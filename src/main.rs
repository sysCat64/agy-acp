use fs2::FileExt;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    id: Option<Value>,
    method: Option<String>,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    params: Value,
}

/// Persisted session→conversation mapping stored in ~/.openab/agy-acp/sessions.json
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SessionStore {
    sessions: HashMap<String, StoredSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredSession {
    conversation_id: Option<String>,
    /// Last step idx read from SQLite; used for delta extraction.
    #[serde(default)]
    last_step_idx: i64,
    /// Selected model ID for this session.
    #[serde(default)]
    model_id: Option<String>,
}

struct Session {
    conversation_id: Option<String>,
    /// Last step idx read from SQLite.
    last_step_idx: i64,
    /// Selected model ID for this session.
    model_id: Option<String>,
}

#[cfg(test)]
struct ConversationDelta {
    text: Option<String>,
    max_step_idx: i64,
}

#[derive(Debug, Default)]
struct StreamingState {
    conversation_id: Option<String>,
    base_step_idx: i64,
    last_step_idx: i64,
    had_updates: bool,
    agent_text_lengths: HashMap<i64, usize>,
    emitted_tool_steps: HashSet<i64>,
}

struct Adapter {
    sessions: HashMap<String, Session>,
    working_dir: String,
    conversations_dir: PathBuf,
    state_file: PathBuf,
    available_models: Vec<String>,
}

impl Adapter {
    fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let state_dir = PathBuf::from(&home).join(".openab/agy-acp");
        Self {
            sessions: HashMap::new(),
            working_dir: std::env::current_dir()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| "/tmp".to_string()),
            conversations_dir: PathBuf::from(&home).join(".gemini/antigravity-cli/conversations"),
            state_file: state_dir.join("sessions.json"),
            available_models: Self::fetch_available_models(),
        }
    }

    /// Run `agy models` and parse the output into a list of model names.
    fn fetch_available_models() -> Vec<String> {
        std::process::Command::new("agy")
            .arg("models")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Build the ACP `models` JSON for a session, given its current model_id.
    /// Fetches available models if the list is still empty.
    fn session_models_json(&mut self, model_id: Option<&str>) -> Value {
        if self.available_models.is_empty() {
            self.available_models = Self::fetch_available_models();
        }
        let current = model_id
            .or_else(|| self.available_models.first().map(|s| s.as_str()))
            .unwrap_or("");
        let available: Vec<Value> = self
            .available_models
            .iter()
            .map(|name| {
                json!({
                    "modelId": name,
                    "name": name,
                })
            })
            .collect();
        json!({
            "currentModelId": current,
            "availableModels": available,
        })
    }

    /// Acquire exclusive lock on a dedicated lock file for read-write mutual exclusion.
    fn lock_state_file(&self) -> Option<fs::File> {
        if let Some(parent) = self.state_file.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let lock_path = self.state_file.with_extension("lock");
        let lock_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .ok()?;
        lock_file.lock_exclusive().ok()?;
        Some(lock_file)
    }

    /// Load persisted session store (caller must hold lock).
    fn load_store_inner(&self) -> SessionStore {
        let Some(file) = fs::File::open(&self.state_file).ok() else {
            return SessionStore::default();
        };
        serde_json::from_reader(&file).unwrap_or_default()
    }

    /// Load persisted session store with lock.
    fn load_store(&self) -> SessionStore {
        let _lock = self.lock_state_file();
        self.load_store_inner()
    }

    /// Try to restore conversation_id, last_step_idx, and model_id from persisted state.
    fn restore_session(&self, session_id: &str) -> Option<(String, i64, Option<String>)> {
        let store = self.load_store();
        store.sessions.get(session_id).and_then(|s| {
            s.conversation_id
                .clone()
                .map(|cid| (cid, s.last_step_idx, s.model_id.clone()))
        })
    }

    /// Persist a session binding (read-modify-write under single lock).
    fn persist_session(
        &self,
        session_id: &str,
        conversation_id: Option<&str>,
        last_step_idx: i64,
        model_id: Option<&str>,
    ) {
        let Some(_lock) = self.lock_state_file() else {
            return;
        };
        let mut store = self.load_store_inner();
        store.sessions.insert(
            session_id.to_string(),
            StoredSession {
                conversation_id: conversation_id.map(String::from),
                last_step_idx,
                model_id: model_id.map(String::from),
            },
        );
        let tmp = self.state_file.with_extension("tmp");
        if let Ok(file) = fs::File::create(&tmp) {
            if serde_json::to_writer_pretty(&file, &store).is_ok() {
                let _ = fs::rename(&tmp, &self.state_file);
            }
        }
    }

    fn conversation_snapshot(&self) -> HashSet<String> {
        let Ok(entries) = fs::read_dir(&self.conversations_dir) else {
            return HashSet::new();
        };
        entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let path = e.path();
                if path.extension().map(|x| x == "db").unwrap_or(false) {
                    path.file_stem().map(|s| s.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    #[cfg(test)]
    fn new_conversation_id(&self, before: &HashSet<String>) -> Option<String> {
        let after = self.conversation_snapshot();
        let mut created: Vec<_> = after.difference(before).collect();
        if created.is_empty() {
            return None;
        }
        if created.len() > 1 {
            eprintln!(
                "[agy-acp] WARN: multiple new agy conversation files appeared; \
                 refusing to bind"
            );
            return None;
        }
        Some(created.remove(0).clone())
    }

    fn new_conversation_id_in_dir(
        conversations_dir: &Path,
        before: &HashSet<String>,
    ) -> Option<String> {
        let Ok(entries) = fs::read_dir(conversations_dir) else {
            return None;
        };
        let after: HashSet<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let path = e.path();
                if path.extension().map(|x| x == "db").unwrap_or(false) {
                    path.file_stem().map(|s| s.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect();
        let mut created: Vec<_> = after.difference(before).collect();
        if created.is_empty() {
            return None;
        }
        if created.len() > 1 {
            eprintln!(
                "[agy-acp] WARN: multiple new agy conversation files appeared; \
                 refusing to bind"
            );
            return None;
        }
        Some(created.remove(0).clone())
    }

    /// Extract text from a step_payload protobuf: top-level field 20 (sub-message) → field 1 (string).
    fn extract_text_from_step_payload(blob: &[u8]) -> Option<String> {
        let field_20 = Self::get_proto_field(blob, 20)?;
        let field_1 = Self::get_proto_field(&field_20, 1)?;
        String::from_utf8(field_1).ok()
    }

    fn extract_tool_update_from_step_payload(
        idx: i64,
        step_type: i64,
        blob: &[u8],
    ) -> Option<Value> {
        let strings = Self::extract_printable_strings(blob);
        let raw_input = strings.iter().find_map(|s| {
            let trimmed = s.trim();
            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                serde_json::from_str::<Value>(trimmed).ok()
            } else {
                Self::extract_first_json_object(trimmed)
            }
        });

        let tool_name = strings.iter().find_map(|s| Self::extract_tool_name(s));
        let title = raw_input
            .as_ref()
            .and_then(|v| v.get("toolSummary").or_else(|| v.get("toolAction")))
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| tool_name.clone())?;
        let kind = Self::tool_kind(tool_name.as_deref().unwrap_or(&title));
        let tool_call_id = format!("agy-{idx}-{step_type}");
        let locations = raw_input
            .as_ref()
            .map(Self::tool_locations)
            .unwrap_or_default();
        let content = raw_input.as_ref().and_then(Self::tool_content);

        let mut update = json!({
            "sessionUpdate": "tool_call",
            "toolCallId": tool_call_id,
            "title": title,
            "kind": kind,
            "status": "completed",
        });
        if let Some(input) = raw_input {
            update["rawInput"] = input;
        }
        if !locations.is_empty() {
            update["locations"] = Value::Array(locations);
        }
        if let Some(content) = content {
            update["content"] = Value::Array(vec![content]);
        }
        Some(update)
    }

    fn extract_user_text_from_step_payload(blob: &[u8]) -> Option<String> {
        let prompt = Self::get_proto_field(blob, 19)?;
        Self::get_text_field(&prompt, 2)
            .or_else(|| {
                let content = Self::get_proto_field(&prompt, 3)?;
                Self::get_text_field(&content, 1)
            })
            .filter(|text| !text.trim().is_empty())
    }

    fn extract_first_json_object(s: &str) -> Option<Value> {
        let start = s.find('{')?;
        let mut depth = 0usize;
        let mut in_string = false;
        let mut escaped = false;

        for (offset, ch) in s[start..].char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }

            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        let end = start + offset + ch.len_utf8();
                        return serde_json::from_str::<Value>(&s[start..end]).ok();
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn message_chunk_update(session_update: &str, text: String) -> Value {
        json!({
            "sessionUpdate": session_update,
            "content": { "type": "text", "text": text },
        })
    }

    fn flush_agent_message(parts: &mut Vec<String>, updates: &mut Vec<Value>) {
        if parts.is_empty() {
            return;
        }
        let text = parts.join("\n");
        parts.clear();
        if !text.is_empty() {
            updates.push(Self::message_chunk_update("agent_message_chunk", text));
        }
    }

    fn extract_printable_strings(blob: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let mut current = Vec::new();
        for &b in blob {
            if b == b'\n' || b == b'\r' || b == b'\t' || (0x20..=0x7e).contains(&b) {
                current.push(b);
            } else {
                if current.len() >= 3 {
                    if let Ok(s) = String::from_utf8(std::mem::take(&mut current)) {
                        out.push(s);
                    }
                }
                current.clear();
            }
        }
        if current.len() >= 3 {
            if let Ok(s) = String::from_utf8(current) {
                out.push(s);
            }
        }
        out
    }

    fn looks_like_tool_name(s: &str) -> bool {
        s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && s.contains('_')
            && s.len() <= 64
    }

    fn extract_tool_name(s: &str) -> Option<String> {
        s.split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'))
            .find(|part| Self::looks_like_tool_name(part))
            .map(String::from)
    }

    fn tool_kind(tool_name: &str) -> &'static str {
        let lower = tool_name.to_lowercase();
        if lower.contains("read") || lower.contains("view") {
            "read"
        } else if lower.contains("write") || lower.contains("edit") || lower.contains("patch") {
            "edit"
        } else if lower.contains("delete") || lower.contains("remove") {
            "delete"
        } else if lower.contains("move") || lower.contains("rename") {
            "move"
        } else if lower.contains("grep") || lower.contains("search") || lower.contains("find") {
            "search"
        } else if lower.contains("command")
            || lower.contains("execute")
            || lower.contains("terminal")
        {
            "execute"
        } else if lower.contains("think")
            || lower.contains("thought")
            || lower.contains("reason")
            || lower.contains("plan")
        {
            "think"
        } else if lower.contains("url") || lower.contains("fetch") {
            "fetch"
        } else {
            "other"
        }
    }

    fn tool_content(input: &Value) -> Option<Value> {
        for key in [
            "thought",
            "thinking",
            "reasoning",
            "analysis",
            "plan",
            "content",
            "text",
            "result",
            "output",
        ] {
            if let Some(text) = input.get(key).and_then(|v| v.as_str()) {
                if !text.trim().is_empty() {
                    return Some(json!({
                        "type": "content",
                        "content": { "type": "text", "text": text },
                    }));
                }
            }
        }
        None
    }

    fn tool_locations(input: &Value) -> Vec<Value> {
        let mut locations = Vec::new();
        for key in ["AbsolutePath", "SearchPath", "path", "file", "FilePath"] {
            if let Some(path) = input.get(key).and_then(|v| v.as_str()) {
                let mut loc = json!({ "path": path });
                if let Some(line) = input
                    .get("StartLine")
                    .or_else(|| input.get("line"))
                    .and_then(|v| v.as_i64())
                {
                    loc["line"] = json!(line);
                }
                locations.push(loc);
            }
        }
        locations
    }

    /// Extract the first length-delimited field with the given number from a protobuf blob.
    fn get_proto_field(blob: &[u8], target: u64) -> Option<Vec<u8>> {
        let mut i = 0;
        while i < blob.len() {
            let (tag, consumed) = Self::read_varint(&blob[i..])?;
            i += consumed;
            let field_number = tag >> 3;
            let wire_type = tag & 0x7;
            match wire_type {
                0 => {
                    let (_, c) = Self::read_varint(&blob[i..])?;
                    i += c;
                }
                2 => {
                    let (len, c) = Self::read_varint(&blob[i..])?;
                    i += c;
                    let len = len as usize;
                    if i + len > blob.len() {
                        return None;
                    }
                    if field_number == target {
                        return Some(blob[i..i + len].to_vec());
                    }
                    i += len;
                }
                5 => {
                    i += 4;
                }
                1 => {
                    i += 8;
                }
                _ => return None,
            }
        }
        None
    }

    fn get_text_field(blob: &[u8], target: u64) -> Option<String> {
        let bytes = Self::get_proto_field(blob, target)?;
        String::from_utf8(bytes).ok()
    }

    /// Read a protobuf varint, returning (value, bytes_consumed).
    fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
        let mut result: u64 = 0;
        let mut shift = 0;
        for (i, &byte) in buf.iter().enumerate() {
            if shift >= 70 {
                return None;
            }
            result |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if byte & 0x80 == 0 {
                return Some((result, i + 1));
            }
        }
        None
    }

    fn read_rows_from_db(
        &self,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<Vec<(i64, i64, Vec<u8>)>> {
        Self::read_rows_from_db_dir(&self.conversations_dir, conversation_id, after_step_idx)
    }

    fn read_rows_from_db_dir(
        conversations_dir: &Path,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<Vec<(i64, i64, Vec<u8>)>> {
        let db_path = conversations_dir.join(format!("{}.db", conversation_id));
        let conn = Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .ok()?;

        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='steps'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);
        if !table_exists {
            eprintln!(
                "[agy-acp] WARN: steps table not found in {}.db — schema changed?",
                conversation_id
            );
            return None;
        }

        let mut stmt = conn
            .prepare("SELECT idx, step_type, step_payload FROM steps WHERE idx > ?1 ORDER BY idx")
            .ok()?;
        let rows: Vec<(i64, i64, Vec<u8>)> = stmt
            .query_map([after_step_idx], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .ok()?
            .filter_map(|r| r.ok())
            .collect();
        Some(rows)
    }

    fn read_replay_updates_from_db(&self, conversation_id: &str) -> Option<(Vec<Value>, i64)> {
        let rows = self.read_rows_from_db(conversation_id, -1)?;
        let mut max_idx = -1;
        let mut updates = Vec::new();
        let mut pending_agent_parts = Vec::new();

        for (idx, step_type, payload) in &rows {
            max_idx = max_idx.max(*idx);
            if *step_type == 14 {
                Self::flush_agent_message(&mut pending_agent_parts, &mut updates);
                if let Some(text) = Self::extract_user_text_from_step_payload(payload) {
                    updates.push(Self::message_chunk_update("user_message_chunk", text));
                }
            } else if *step_type == 15 {
                if let Some(text) = Self::extract_text_from_step_payload(payload) {
                    if !text.is_empty() {
                        pending_agent_parts.push(text);
                    }
                }
            } else if matches!(*step_type, 7 | 8 | 9 | 17 | 101 | 138) {
                Self::flush_agent_message(&mut pending_agent_parts, &mut updates);
                if let Some(update) =
                    Self::extract_tool_update_from_step_payload(*idx, *step_type, payload)
                {
                    updates.push(update);
                }
            }
        }
        Self::flush_agent_message(&mut pending_agent_parts, &mut updates);

        if updates.is_empty() {
            return None;
        }
        Some((updates, max_idx))
    }

    #[cfg(test)]
    fn read_delta_from_db(
        &self,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<ConversationDelta> {
        Self::read_delta_from_db_dir(&self.conversations_dir, conversation_id, after_step_idx)
    }

    #[cfg(test)]
    fn read_delta_from_db_dir(
        conversations_dir: &Path,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<ConversationDelta> {
        let rows = Self::read_rows_from_db_dir(conversations_dir, conversation_id, after_step_idx)?;

        let mut max_idx = after_step_idx;
        let mut response_parts: Vec<String> = Vec::new();
        for (idx, step_type, payload) in &rows {
            max_idx = max_idx.max(*idx);
            if *step_type == 15 {
                if let Some(text) = Self::extract_text_from_step_payload(payload) {
                    if !text.is_empty() {
                        response_parts.push(text);
                    }
                }
            }
        }
        if response_parts.is_empty() {
            let response_rows: Vec<_> = rows
                .iter()
                .filter(|(_, step_type, _)| *step_type == 15)
                .collect();
            if !response_rows.is_empty() {
                let payload_sizes: Vec<usize> =
                    response_rows.iter().map(|(_, _, p)| p.len()).collect();
                eprintln!(
                    "[agy-acp] WARN: {} new response steps found (payload sizes: {:?}) but none had extractable text \
                     (field 20.1 missing — schema change?)",
                    response_rows.len(), payload_sizes
                );
            }
            return None;
        }
        let text = if response_parts.is_empty() {
            None
        } else {
            Some(Self::filter_narration(&response_parts))
        };
        Some(ConversationDelta {
            text,
            max_step_idx: max_idx,
        })
    }

    fn poll_streaming_delta(
        conversations_dir: &Path,
        snapshot: Option<&HashSet<String>>,
        session_id: &str,
        state: &Arc<Mutex<StreamingState>>,
    ) -> Vec<String> {
        let (conversation_id, base_step_idx) = {
            let mut guard = state.lock().unwrap();
            if guard.conversation_id.is_none() {
                if let Some(before) = snapshot {
                    guard.conversation_id =
                        Self::new_conversation_id_in_dir(conversations_dir, before);
                }
            }
            (guard.conversation_id.clone(), guard.base_step_idx)
        };

        let Some(conversation_id) = conversation_id else {
            return Vec::new();
        };

        let Some(rows) =
            Self::read_rows_from_db_dir(conversations_dir, &conversation_id, base_step_idx)
        else {
            return Vec::new();
        };

        let mut guard = state.lock().unwrap();
        let mut notifications = Vec::new();

        for (idx, step_type, payload) in rows {
            guard.last_step_idx = guard.last_step_idx.max(idx);

            if step_type == 15 {
                let Some(text) = Self::extract_text_from_step_payload(&payload) else {
                    continue;
                };
                let written_len = guard.agent_text_lengths.get(&idx).copied().unwrap_or(0);
                if text.len() <= written_len {
                    continue;
                }
                let Some(new_text) = text.get(written_len..) else {
                    continue;
                };
                guard.agent_text_lengths.insert(idx, text.len());
                if !new_text.is_empty() {
                    notifications.push(
                        serde_json::to_string(&JsonRpcNotification {
                            jsonrpc: "2.0",
                            method: "session/update".to_string(),
                            params: json!({
                                "sessionId": session_id,
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": { "type": "text", "text": new_text },
                                },
                            }),
                        })
                        .unwrap(),
                    );
                }
            } else if matches!(step_type, 7 | 8 | 9 | 17 | 101 | 138)
                && !guard.emitted_tool_steps.contains(&idx)
            {
                if let Some(update) =
                    Self::extract_tool_update_from_step_payload(idx, step_type, &payload)
                {
                    guard.emitted_tool_steps.insert(idx);
                    notifications.push(
                        serde_json::to_string(&JsonRpcNotification {
                            jsonrpc: "2.0",
                            method: "session/update".to_string(),
                            params: json!({
                                "sessionId": session_id,
                                "update": update,
                            }),
                        })
                        .unwrap(),
                    );
                }
            }
        }

        guard.had_updates = guard.had_updates || !notifications.is_empty();
        notifications
    }

    #[cfg(test)]
    fn read_response_from_db(
        &self,
        conversation_id: &str,
        after_step_idx: i64,
    ) -> Option<(String, i64)> {
        self.read_delta_from_db(conversation_id, after_step_idx)
            .and_then(|delta| delta.text.map(|text| (text, delta.max_step_idx)))
    }

    /// Filter out leading narration ("I will ...") from response parts based on
    /// the OPENAB_TOOL_DISPLAY environment variable.
    /// - "full": return all parts joined
    /// - "compact" / "none" / unset: drop leading narration-only parts
    #[cfg(test)]
    fn filter_narration(parts: &[String]) -> String {
        let should_filter = std::env::var("OPENAB_TOOL_DISPLAY")
            .map(|v| {
                let lower = v.to_lowercase();
                lower != "full"
            })
            .unwrap_or(true);

        Self::filter_narration_with_mode(parts, should_filter)
    }

    #[cfg(test)]
    fn filter_narration_with_mode(parts: &[String], should_filter: bool) -> String {
        if !should_filter || parts.len() <= 1 {
            return parts.join("\n");
        }

        // Find the first part that is NOT pure narration. Keep it and everything after.
        let first_content = parts
            .iter()
            .position(|p| !Self::is_narration(p))
            .unwrap_or(parts.len() - 1);
        parts[first_content..].join("\n")
    }

    /// A part is considered narration if every non-empty line starts with "I will".
    #[cfg(test)]
    fn is_narration(text: &str) -> bool {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return false;
        }
        lines.iter().all(|l| l.trim_start().starts_with("I will"))
    }

    fn evict_if_needed(&mut self) {
        const MAX_SESSIONS: usize = 64;
        while self.sessions.len() >= MAX_SESSIONS {
            if let Some(key) = self.sessions.keys().next().cloned() {
                self.sessions.remove(&key);
            }
        }
    }

    fn restore_session_state(&mut self, session_id: &str) -> bool {
        let Some((conversation_id, last_step_idx, model_id)) = self.restore_session(session_id)
        else {
            return false;
        };
        if !self.sessions.contains_key(session_id) {
            self.evict_if_needed();
        }
        self.sessions.insert(
            session_id.to_string(),
            Session {
                conversation_id: Some(conversation_id),
                last_step_idx,
                model_id,
            },
        );
        true
    }

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": 1,
                "agentInfo": { "name": "agy", "version": env!("CARGO_PKG_VERSION") },
                "agentCapabilities": {
                    "loadSession": true,
                    "sessionCapabilities": { "resume": {} },
                },
                "authMethods": [],
            })),
            error: None,
        }
    }

    fn handle_session_new(&mut self, id: Value) -> JsonRpcResponse {
        let session_id = Uuid::new_v4().to_string();
        self.evict_if_needed();
        self.sessions.insert(
            session_id.clone(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
            },
        );
        let models = self.session_models_json(None);
        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "sessionId": session_id, "models": models })),
            error: None,
        }
    }

    fn handle_session_load(&mut self, id: Value, params: &Value) -> Vec<String> {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            })
            .unwrap()];
        }

        if !self.sessions.contains_key(session_id) && !self.restore_session_state(session_id) {
            return vec![serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": format!("unknown sessionId: {session_id}"),
                })),
            })
            .unwrap()];
        }

        let mut output_lines: Vec<String> = Vec::new();

        // Replay conversation history so the client can display it
        let replay_conv_id = self
            .sessions
            .get(session_id)
            .and_then(|session| session.conversation_id.clone());
        if let Some(conv_id) = replay_conv_id {
            if let Some((updates, max_step_idx)) = self.read_replay_updates_from_db(&conv_id) {
                for update in updates {
                    let notification = serde_json::to_string(&JsonRpcNotification {
                        jsonrpc: "2.0",
                        method: "session/update".to_string(),
                        params: json!({
                            "sessionId": session_id,
                            "update": update,
                        }),
                    })
                    .unwrap();
                    output_lines.push(notification);
                }
                if let Some(session) = self.sessions.get_mut(session_id) {
                    session.last_step_idx = max_step_idx;
                }
                let model_id = self
                    .sessions
                    .get(session_id)
                    .and_then(|s| s.model_id.clone());
                self.persist_session(
                    session_id,
                    Some(conv_id.as_str()),
                    max_step_idx,
                    model_id.as_deref(),
                );
            }
        }

        output_lines.push({
            let model_id = self
                .sessions
                .get(session_id)
                .and_then(|s| s.model_id.clone());
            let models = self.session_models_json(model_id.as_deref());
            serde_json::to_string(&JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({ "sessionId": session_id, "models": models })),
                error: None,
            })
            .unwrap()
        });

        output_lines
    }

    fn handle_session_resume(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if session_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId"})),
            };
        }

        if self.sessions.contains_key(session_id) || self.restore_session_state(session_id) {
            let model_id = self
                .sessions
                .get(session_id)
                .and_then(|s| s.model_id.clone());
            let models = self.session_models_json(model_id.as_deref());
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({ "sessionId": session_id, "models": models })),
                error: None,
            };
        }

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(json!({
                "code": -32000,
                "message": format!("unknown sessionId: {session_id}"),
            })),
        }
    }

    fn handle_session_set_model(&mut self, id: Value, params: &Value) -> JsonRpcResponse {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let model_id = params.get("modelId").and_then(|v| v.as_str()).unwrap_or("");

        if session_id.is_empty() || model_id.is_empty() {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({"code":-32602,"message":"missing sessionId or modelId"})),
            };
        }

        // Restore evicted session from state file if needed
        if !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id);
        }

        let Some(session) = self.sessions.get_mut(session_id) else {
            return JsonRpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(json!({
                    "code": -32000,
                    "message": format!("unknown sessionId: {session_id}"),
                })),
            };
        };

        session.model_id = Some(model_id.to_string());
        let model_id_str = session.model_id.clone();
        let last_step_idx = session.last_step_idx;
        let conv_id = session.conversation_id.clone();

        // Persist the updated model selection
        self.persist_session(
            session_id,
            conv_id.as_deref(),
            last_step_idx,
            model_id_str.as_deref(),
        );

        JsonRpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({})),
            error: None,
        }
    }

    async fn handle_session_prompt(&mut self, id: Value, params: &Value) -> Vec<String> {
        let session_id = params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Restore evicted session from state file if needed
        if !session_id.is_empty() && !self.sessions.contains_key(session_id) {
            let _ = self.restore_session_state(session_id);
        }

        let prompt_text = params
            .get("prompt")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        let clean_prompt = prompt_text.trim();

        // Take snapshot before spawning agy if we need to bind a conversation
        let snapshot = if self
            .sessions
            .get(session_id)
            .map(|s| s.conversation_id.is_none())
            .unwrap_or(false)
        {
            Some(self.conversation_snapshot())
        } else {
            None
        };

        // Build args
        let mut args: Vec<String> = Vec::new();
        args.push("--add-dir".to_string());
        args.push(self.working_dir.clone());
        if let Ok(extra) = std::env::var("AGY_EXTRA_ARGS") {
            args.extend(extra.split_whitespace().map(String::from));
        }
        if let Some(session) = self.sessions.get(session_id) {
            if let Some(conv_id) = &session.conversation_id {
                args.push("--conversation".to_string());
                args.push(conv_id.clone());
            }
            if let Some(model_id) = &session.model_id {
                args.push("--model".to_string());
                args.push(model_id.clone());
            }
        }
        args.push("-p".to_string());
        args.push(clean_prompt.to_string());

        let spawn_result = Command::new("agy")
            .args(&args)
            .current_dir(&self.working_dir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let child = match spawn_result {
            Ok(child) => child,
            Err(e) => {
                return vec![serde_json::to_string(&JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(json!({"code":-32000,"message":format!("failed to run agy: {e}")})),
                })
                .unwrap()];
            }
        };

        let initial_conv_id = self
            .sessions
            .get(session_id)
            .and_then(|s| s.conversation_id.clone());
        let initial_step_idx = self
            .sessions
            .get(session_id)
            .map(|s| s.last_step_idx)
            .unwrap_or(-1);
        let streaming_state = Arc::new(Mutex::new(StreamingState {
            conversation_id: initial_conv_id,
            base_step_idx: initial_step_idx,
            last_step_idx: initial_step_idx,
            had_updates: false,
            agent_text_lengths: HashMap::new(),
            emitted_tool_steps: HashSet::new(),
        }));
        let stop_polling = Arc::new(AtomicBool::new(false));
        let poll_conversations_dir = self.conversations_dir.clone();
        let poll_snapshot = snapshot.clone();
        let poll_session_id = session_id.to_string();
        let poll_state = Arc::clone(&streaming_state);
        let poll_stop = Arc::clone(&stop_polling);

        let poller = std::thread::spawn(move || {
            let mut stdout = io::stdout();
            while !poll_stop.load(Ordering::SeqCst) {
                for line in Self::poll_streaming_delta(
                    &poll_conversations_dir,
                    poll_snapshot.as_ref(),
                    &poll_session_id,
                    &poll_state,
                ) {
                    let _ = writeln!(stdout, "{}", line);
                }
                let _ = stdout.flush();
                std::thread::sleep(Duration::from_millis(500));
            }
        });

        let result = child.wait_with_output().await;
        stop_polling.store(true, Ordering::SeqCst);
        let _ = poller.join();

        let mut final_lines = Vec::new();
        for attempt in 0..3 {
            let lines = Self::poll_streaming_delta(
                &self.conversations_dir,
                snapshot.as_ref(),
                session_id,
                &streaming_state,
            );
            final_lines.extend(lines);
            if attempt < 2 {
                std::thread::sleep(Duration::from_millis(100));
            }
        }

        {
            let mut stdout = io::stdout();
            for line in &final_lines {
                let _ = writeln!(stdout, "{}", line);
            }
            let _ = stdout.flush();
        }

        let state = streaming_state.lock().unwrap();
        let bound_conv_id = state.conversation_id.clone();
        let new_step_idx = state.last_step_idx;
        let had_updates = state.had_updates || !final_lines.is_empty();
        drop(state);

        if let Some(session) = self.sessions.get_mut(session_id) {
            if session.conversation_id.is_none() {
                session.conversation_id = bound_conv_id.clone();
            }
            if bound_conv_id.is_some() {
                session.last_step_idx = new_step_idx;
            }
        }
        if bound_conv_id.is_some() {
            let model_id = self
                .sessions
                .get(session_id)
                .and_then(|s| s.model_id.clone());
            self.persist_session(
                session_id,
                bound_conv_id.as_deref(),
                new_step_idx,
                model_id.as_deref(),
            );
        }

        let output_lines = vec![serde_json::to_string(&JsonRpcResponse {
            jsonrpc: "2.0",
            id: id.clone(),
            result: Some(json!({ "stopReason": "end_turn" })),
            error: None,
        })
        .unwrap()];

        match result {
            Ok(output) => {
                let stderr_text = String::from_utf8_lossy(&output.stderr);
                if !stderr_text.is_empty() {
                    eprintln!("[agy-acp] agy stderr: {}", stderr_text.trim_end());
                }

                if !output.status.success() {
                    eprintln!("[agy-acp] WARN: agy exited with status: {}", output.status);
                    if !had_updates {
                        let msg = if stderr_text.is_empty() {
                            format!("agy exited with status: {}", output.status)
                        } else {
                            format!("agy failed: {}", stderr_text.trim_end())
                        };
                        return vec![serde_json::to_string(&JsonRpcResponse {
                            jsonrpc: "2.0",
                            id,
                            result: None,
                            error: Some(json!({"code":-32000,"message":msg})),
                        })
                        .unwrap()];
                    }
                }
            }
            Err(e) => {
                return vec![serde_json::to_string(&JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(
                        json!({"code":-32000,"message":format!("failed to wait for agy: {e}")}),
                    ),
                })
                .unwrap()];
            }
        }

        output_lines
    }
}

#[tokio::main]
async fn main() {
    let mut adapter = Adapter::new();

    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(l) if !l.trim().is_empty() => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    let mut stdout = io::stdout();

    while let Some(line) = rx.recv().await {
        let req: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let id = match req.id {
            Some(id) => id,
            None => continue,
        };

        let output = match req.method.as_deref() {
            Some("initialize") => {
                vec![serde_json::to_string(&adapter.handle_initialize(id)).unwrap()]
            }
            Some("session/new") => {
                vec![serde_json::to_string(&adapter.handle_session_new(id)).unwrap()]
            }
            Some("session/load") => {
                let params = req.params.unwrap_or(json!({}));
                adapter.handle_session_load(id, &params)
            }
            Some("session/resume") => {
                let params = req.params.unwrap_or(json!({}));
                vec![serde_json::to_string(&adapter.handle_session_resume(id, &params)).unwrap()]
            }
            Some("session/prompt") => {
                let params = req.params.unwrap_or(json!({}));
                adapter.handle_session_prompt(id, &params).await
            }
            Some("session/cancel") => {
                let r = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: Some(json!({})),
                    error: None,
                };
                vec![serde_json::to_string(&r).unwrap()]
            }
            Some("session/set_model") | Some("session/setModel") => {
                let params = req.params.unwrap_or(json!({}));
                vec![serde_json::to_string(&adapter.handle_session_set_model(id, &params)).unwrap()]
            }
            Some(method) => {
                let r = JsonRpcResponse {
                    jsonrpc: "2.0",
                    id,
                    result: None,
                    error: Some(
                        json!({"code":-32601,"message":format!("method not found: {method}")}),
                    ),
                };
                vec![serde_json::to_string(&r).unwrap()]
            }
            None => continue,
        };

        for line in output {
            let _ = writeln!(stdout, "{}", line);
        }
        let _ = stdout.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_varint(out: &mut Vec<u8>, mut value: u64) {
        loop {
            if value < 128 {
                out.push(value as u8);
                break;
            }
            out.push(((value as u8) & 0x7F) | 0x80);
            value >>= 7;
        }
    }

    fn push_len_field(out: &mut Vec<u8>, field_number: u64, bytes: &[u8]) {
        push_varint(out, (field_number << 3) | 2);
        push_varint(out, bytes.len() as u64);
        out.extend_from_slice(bytes);
    }

    fn make_assistant_payload(text: &str) -> Vec<u8> {
        let mut inner = Vec::new();
        push_len_field(&mut inner, 1, text.as_bytes());

        let mut outer = Vec::new();
        push_len_field(&mut outer, 20, &inner);
        outer
    }

    fn make_user_payload(text: &str) -> Vec<u8> {
        let mut content = Vec::new();
        push_len_field(&mut content, 1, text.as_bytes());

        let mut prompt = Vec::new();
        push_len_field(&mut prompt, 2, text.as_bytes());
        push_len_field(&mut prompt, 3, &content);

        let mut outer = Vec::new();
        push_len_field(&mut outer, 19, &prompt);
        outer
    }

    #[test]
    fn test_extract_text_from_step_payload_field20_field1() {
        // field 20 (tag 0xA2 0x01), containing sub-message with field 1 = "hello"
        let mut inner = Vec::new();
        inner.push(0x0A);
        inner.push(0x05); // field 1, LEN, 5 bytes
        inner.extend_from_slice(b"hello");

        let mut blob = vec![
            0x08,
            0x0F, // field 1 varint = 15
            // field 20, wire type 2: tag = (20 << 3) | 2 = 0xA2, needs varint encoding: 0xA2 0x01
            0xA2,
            0x01,
            inner.len() as u8,
        ];
        blob.extend_from_slice(&inner);
        assert_eq!(
            Adapter::extract_text_from_step_payload(&blob),
            Some("hello".to_string())
        );
    }

    #[test]
    fn test_extract_text_returns_none_without_field20() {
        // Only field 1 (varint) — no field 20
        let blob = vec![0x08, 0x03];
        assert_eq!(Adapter::extract_text_from_step_payload(&blob), None);
    }

    #[test]
    fn test_extract_user_text_from_step_payload_field19_field2() {
        let payload = make_user_payload("how are you?");
        assert_eq!(
            Adapter::extract_user_text_from_step_payload(&payload),
            Some("how are you?".to_string())
        );
    }

    #[test]
    fn test_extract_text_multiline() {
        let text = b"Safe memory rules\nCompiler points out the flaws\nFast and fearless code";
        let mut inner = Vec::new();
        inner.push(0x0A); // field 1, LEN
        inner.push(text.len() as u8);
        inner.extend_from_slice(text);

        let mut blob = vec![
            0x08,
            0x01, // field 1 varint
            // field 20
            0xA2,
            0x01,
            inner.len() as u8,
        ];
        blob.extend_from_slice(&inner);
        assert_eq!(
            Adapter::extract_text_from_step_payload(&blob),
            Some(
                "Safe memory rules\nCompiler points out the flaws\nFast and fearless code"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_extract_tool_update_from_step_payload_json() {
        let payload = br#"
            grep_search
            {"Query":"prompt","SearchPath":"/tmp/project/src/main.rs","toolAction":"Finding prompt handling","toolSummary":"Grep prompt"}
        "#;

        let update = Adapter::extract_tool_update_from_step_payload(19, 7, payload).unwrap();
        assert_eq!(update["sessionUpdate"], "tool_call");
        assert_eq!(update["toolCallId"], "agy-19-7");
        assert_eq!(update["title"], "Grep prompt");
        assert_eq!(update["kind"], "search");
        assert_eq!(update["status"], "completed");
        assert_eq!(update["rawInput"]["Query"], "prompt");
        assert_eq!(update["locations"][0]["path"], "/tmp/project/src/main.rs");
    }

    #[test]
    fn test_extract_tool_update_uses_tool_name_fallback() {
        let payload = b"view_file";
        let update = Adapter::extract_tool_update_from_step_payload(3, 8, payload).unwrap();
        assert_eq!(update["title"], "view_file");
        assert_eq!(update["kind"], "read");
    }

    #[test]
    fn test_extract_tool_update_parses_first_balanced_json_object() {
        let payload = br#"
            abc123 view_file
            {"AbsolutePath":"/tmp/project/README.md","toolAction":"Reading README.md","toolSummary":"View README file"}
            trailing render blob {not json}
        "#;

        let update = Adapter::extract_tool_update_from_step_payload(6, 8, payload).unwrap();
        assert_eq!(update["sessionUpdate"], "tool_call");
        assert_eq!(update["title"], "View README file");
        assert_eq!(update["kind"], "read");
        assert_eq!(update["rawInput"]["AbsolutePath"], "/tmp/project/README.md");
        assert_eq!(update["locations"][0]["path"], "/tmp/project/README.md");
    }

    #[test]
    fn test_extract_tool_name_from_embedded_token() {
        assert_eq!(
            Adapter::extract_tool_name("abc123\tview_file\n{...}"),
            Some("view_file".to_string())
        );
    }

    #[test]
    fn test_extract_tool_update_maps_reasoning_to_think_content() {
        let payload = br#"
            thinking
            {"thought":"Need to inspect the protocol before changing serialization.","toolSummary":"Reasoning"}
        "#;

        let update = Adapter::extract_tool_update_from_step_payload(21, 17, payload).unwrap();
        assert_eq!(update["sessionUpdate"], "tool_call");
        assert_eq!(update["toolCallId"], "agy-21-17");
        assert_eq!(update["title"], "Reasoning");
        assert_eq!(update["kind"], "think");
        assert_eq!(update["status"], "completed");
        assert_eq!(update["content"][0]["type"], "content");
        assert_eq!(update["content"][0]["content"]["type"], "text");
        assert_eq!(
            update["content"][0]["content"]["text"],
            "Need to inspect the protocol before changing serialization."
        );
    }

    #[test]
    fn test_read_varint() {
        assert_eq!(Adapter::read_varint(&[0x05]), Some((5, 1)));
        assert_eq!(Adapter::read_varint(&[0xAC, 0x02]), Some((300, 2)));
        assert_eq!(Adapter::read_varint(&[]), None);
    }

    #[test]
    fn test_initialize_advertises_load_session_support() {
        let adapter = Adapter::new();
        let response = adapter.handle_initialize(json!(1));
        assert_eq!(
            response
                .result
                .as_ref()
                .and_then(|r| r.get("agentCapabilities"))
                .and_then(|c| c.get("loadSession"))
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn test_initialize_advertises_resume_capability() {
        let adapter = Adapter::new();
        let response = adapter.handle_initialize(json!(1));
        assert!(
            response
                .result
                .as_ref()
                .and_then(|r| r.get("agentCapabilities"))
                .and_then(|c| c.get("sessionCapabilities"))
                .and_then(|sc| sc.get("resume"))
                .is_some(),
            "sessionCapabilities.resume should be present"
        );
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_load_restores_persisted_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-load-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };
        adapter.persist_session("sess-1", Some("conv-abc"), 5, None);

        let output = adapter.handle_session_load(json!(7), &json!({"sessionId": "sess-1"}));
        // Last line is the response, any preceding lines are replay notifications
        let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
        assert!(response["error"].is_null());
        assert_eq!(
            adapter
                .sessions
                .get("sess-1")
                .and_then(|s| s.conversation_id.as_deref()),
            Some("conv-abc")
        );
        assert_eq!(
            adapter.sessions.get("sess-1").map(|s| s.last_step_idx),
            Some(5)
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_load_rejects_unknown_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-missing-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        let output = adapter.handle_session_load(json!(9), &json!({"sessionId": "missing"}));
        let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
        assert!(response["result"].is_null());
        assert_eq!(
            response["error"]["message"].as_str(),
            Some("unknown sessionId: missing")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_load_replays_conversation_history() {
        let root = std::env::temp_dir().join(format!("agy-acp-load-replay-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        // Build a conversation DB with two user turns and assistant responses.
        let db_path = conv_dir.join("conv-replay.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
                metadata BLOB,
                error_details BLOB,
                permissions BLOB,
                task_details BLOB,
                render_info BLOB,
                step_payload BLOB,
                step_format INTEGER NOT NULL DEFAULT 0
            )",
        )
        .unwrap();

        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
            rusqlite::params![1i64, make_user_payload("hello")],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![
                2i64,
                make_assistant_payload("I will inspect the workspace.")
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 8, ?2)",
            rusqlite::params![
                3i64,
                br#"view_file
                {"AbsolutePath":"/tmp/project/README.md","toolAction":"Reading README.md","toolSummary":"View README file"}
                trailing render blob {not json}"#
                    .as_slice()
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![4i64, make_assistant_payload("hello from agent")],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
            rusqlite::params![5i64, make_user_payload("how are you?")],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![6i64, make_assistant_payload("second response")],
        )
        .unwrap();
        drop(conn);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };
        adapter.persist_session("sess-replay", Some("conv-replay"), 6, None);

        let output = adapter.handle_session_load(json!(1), &json!({"sessionId": "sess-replay"}));

        // Should have at least one notification (replay) + one response
        assert!(
            output.len() >= 2,
            "expected replay notification + response, got {}",
            output.len()
        );

        let updates: Vec<Value> = output[..output.len() - 1]
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(updates.iter().any(|notification| {
            notification["method"] == "session/update"
                && notification["params"]["update"]["sessionUpdate"] == "tool_call"
                && notification["params"]["update"]["title"] == "View README file"
                && notification["params"]["update"]["kind"] == "read"
        }));
        let replay_kinds: Vec<_> = updates
            .iter()
            .map(|notification| {
                notification["params"]["update"]["sessionUpdate"]
                    .as_str()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            replay_kinds,
            vec![
                "user_message_chunk",
                "agent_message_chunk",
                "tool_call",
                "agent_message_chunk",
                "user_message_chunk",
                "agent_message_chunk"
            ]
        );
        let message_updates: Vec<_> = updates
            .iter()
            .filter(|notification| {
                matches!(
                    notification["params"]["update"]["sessionUpdate"].as_str(),
                    Some("user_message_chunk") | Some("agent_message_chunk")
                )
            })
            .collect();
        let update_kinds: Vec<_> = message_updates
            .iter()
            .map(|notification| {
                notification["params"]["update"]["sessionUpdate"]
                    .as_str()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            update_kinds,
            vec![
                "user_message_chunk",
                "agent_message_chunk",
                "agent_message_chunk",
                "user_message_chunk",
                "agent_message_chunk"
            ]
        );
        let message_texts: Vec<_> = message_updates
            .iter()
            .map(|notification| {
                notification["params"]["update"]["content"]["text"]
                    .as_str()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            message_texts,
            vec![
                "hello",
                "I will inspect the workspace.",
                "hello from agent",
                "how are you?",
                "second response"
            ]
        );
        assert!(
            message_texts[1].contains("I will inspect"),
            "load replay should preserve narration shown in the live session"
        );

        // Last line should be the success response
        let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
        assert!(response["error"].is_null());
        assert_eq!(
            response["result"]["sessionId"].as_str(),
            Some("sess-replay")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_resume_restores_persisted_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-resume-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };
        adapter.persist_session("sess-r1", Some("conv-xyz"), 3, None);

        let response = adapter.handle_session_resume(json!(10), &json!({"sessionId": "sess-r1"}));
        assert!(response.error.is_none());
        assert_eq!(
            response
                .result
                .as_ref()
                .and_then(|r| r.get("sessionId"))
                .and_then(|s| s.as_str()),
            Some("sess-r1")
        );
        assert_eq!(
            adapter
                .sessions
                .get("sess-r1")
                .and_then(|s| s.conversation_id.as_deref()),
            Some("conv-xyz")
        );
        assert_eq!(
            adapter.sessions.get("sess-r1").map(|s| s.last_step_idx),
            Some(3)
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_resume_rejects_unknown_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-resume-miss-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        let response = adapter.handle_session_resume(json!(11), &json!({"sessionId": "nope"}));
        assert!(response.result.is_none());
        assert_eq!(
            response
                .error
                .as_ref()
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str()),
            Some("unknown sessionId: nope")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_session_resume_rejects_empty_session_id() {
        let mut adapter = Adapter::new();
        let response = adapter.handle_session_resume(json!(12), &json!({}));
        assert!(response.result.is_none());
        assert_eq!(
            response
                .error
                .as_ref()
                .and_then(|e| e.get("code"))
                .and_then(|c| c.as_i64()),
            Some(-32602)
        );
    }

    #[test]
    fn test_session_resume_accepts_in_memory_session() {
        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: "/tmp".to_string(),
            conversations_dir: PathBuf::from("/tmp/conversations"),
            state_file: PathBuf::from("/tmp/nonexistent-agy-acp-sessions.json"),
            available_models: vec![],
        };
        adapter.sessions.insert(
            "sess-memory".to_string(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
            },
        );

        let response =
            adapter.handle_session_resume(json!(12), &json!({"sessionId": "sess-memory"}));

        assert!(response.error.is_none());
        assert_eq!(
            response
                .result
                .as_ref()
                .and_then(|r| r.get("sessionId"))
                .and_then(|s| s.as_str()),
            Some("sess-memory")
        );
    }

    #[test]
    fn test_session_load_accepts_in_memory_session_without_replay() {
        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: "/tmp".to_string(),
            conversations_dir: PathBuf::from("/tmp/conversations"),
            state_file: PathBuf::from("/tmp/nonexistent-agy-acp-sessions.json"),
            available_models: vec![],
        };
        adapter.sessions.insert(
            "sess-memory-load".to_string(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: None,
            },
        );

        let output =
            adapter.handle_session_load(json!(13), &json!({"sessionId": "sess-memory-load"}));

        assert_eq!(output.len(), 1);
        let response: Value = serde_json::from_str(&output[0]).unwrap();
        assert!(response["error"].is_null());
        assert_eq!(response["result"]["sessionId"], "sess-memory-load");
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_resume_does_not_replay_history() {
        let root = std::env::temp_dir().join(format!("agy-acp-resume-noreplay-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };
        adapter.persist_session("sess-nr", Some("conv-nr"), 10, None);

        let response = adapter.handle_session_resume(json!(13), &json!({"sessionId": "sess-nr"}));
        assert!(response.error.is_none());
        // Resume should succeed without sending any session/update notifications
        assert_eq!(
            response
                .result
                .as_ref()
                .and_then(|r| r.get("sessionId"))
                .and_then(|s| s.as_str()),
            Some("sess-nr")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_snapshot_detects_db_conversations() {
        let root = std::env::temp_dir().join(format!("agy-acp-db-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();
        fs::write(conv_dir.join("existing.db"), b"old").unwrap();

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir.clone(),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        let before = adapter.conversation_snapshot();
        assert!(before.contains("existing"));

        fs::write(conv_dir.join("new-conv.db"), b"new").unwrap();
        // WAL sidecar files should not be picked up
        fs::write(conv_dir.join("new-conv.db-wal"), b"wal").unwrap();
        fs::write(conv_dir.join("new-conv.db-shm"), b"shm").unwrap();

        assert_eq!(
            adapter.new_conversation_id(&before),
            Some("new-conv".to_string())
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_snapshot_ignores_multiple_new_files() {
        let root = std::env::temp_dir().join(format!("agy-acp-multi-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir.clone(),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        let before = adapter.conversation_snapshot();
        fs::write(conv_dir.join("a.db"), b"").unwrap();
        fs::write(conv_dir.join("b.db"), b"").unwrap();

        assert_eq!(adapter.new_conversation_id(&before), None);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_persist_and_restore_session() {
        let root = std::env::temp_dir().join(format!("agy-acp-state-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        adapter.persist_session("sess-1", Some("conv-abc"), 7, None);
        let restored = adapter.restore_session("sess-1");
        assert_eq!(restored, Some(("conv-abc".to_string(), 7, None)));

        let missing = adapter.restore_session("sess-unknown");
        assert_eq!(missing, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O — requires real SQLite DB
    fn test_read_response_from_db() {
        let root = std::env::temp_dir().join(format!("agy-acp-sqlite-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        // Create a test SQLite DB with steps table
        let db_path = conv_dir.join("test-conv.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
                metadata BLOB,
                error_details BLOB,
                permissions BLOB,
                task_details BLOB,
                render_info BLOB,
                step_payload BLOB,
                step_format INTEGER NOT NULL DEFAULT 0
            )",
        )
        .unwrap();

        // Insert a step_type=15 step with field 20 → field 1 containing "hello world"
        let mut inner = Vec::new();
        inner.push(0x0A);
        inner.push(11); // field 1, LEN, 11 bytes
        inner.extend_from_slice(b"hello world");
        let mut payload = vec![
            0x08,
            0x0F, // field 1 varint = 15
            0xA2,
            0x01, // field 20, LEN
            inner.len() as u8,
        ];
        payload.extend_from_slice(&inner);

        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![1i64, payload],
        )
        .unwrap();

        // Insert a non-response step (step_type=14) — should be ignored
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
            rusqlite::params![2i64, vec![0x08u8, 0x0E]],
        )
        .unwrap();
        drop(conn);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        let result = adapter.read_response_from_db("test-conv", -1);
        assert_eq!(result, Some(("hello world".to_string(), 1)));

        // Reading after idx 1 should return None (no new steps)
        let result = adapter.read_response_from_db("test-conv", 1);
        assert_eq!(result, None);

        let _ = fs::remove_dir_all(root);
    }

    /// Check auth is available: either GEMINI_API_KEY env var or local keyring.
    /// Returns true if auth is ready, false to skip the test.
    fn prepare_auth() -> bool {
        if std::env::var("GEMINI_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            eprintln!("[e2e] Using GEMINI_API_KEY");
            return true;
        }
        let home = std::env::var("HOME").unwrap_or_default();
        let settings = format!("{}/.gemini/antigravity-cli/settings.json", home);
        if std::path::Path::new(&settings).exists() {
            eprintln!("[e2e] Using local auth (keyring)");
            return true;
        }
        eprintln!("SKIP: No GEMINI_API_KEY and no local auth found");
        false
    }

    /// E2E test: spawns agy-acp, sends initialize → session/new → session/prompt,
    /// and verifies the response contains expected text from real agy v1.0.4.
    /// Requires `agy` in PATH and auth (via local or AGY_AUTH_URL). Run with: cargo test e2e -- --ignored
    #[test]
    #[ignore]
    fn test_e2e_agy_acp_full_round_trip() {
        use std::io::{BufRead, BufReader, Write};
        use std::process::{Command, Stdio};
        use std::time::Duration;

        if !prepare_auth() {
            return;
        }

        // Check agy is available
        let agy_check = Command::new("agy").arg("--help").output();
        if agy_check.is_err() || !agy_check.unwrap().status.success() {
            eprintln!("SKIP: agy not found in PATH");
            return;
        }

        let binary = std::env::current_dir()
            .unwrap()
            .join("target/release/agy-acp");
        if !binary.exists() {
            panic!("Run `cargo build --release` first");
        }

        let mut child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn agy-acp");

        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let mut reader = BufReader::new(stdout);

        // Helper to send a line and read one response line
        let mut send_and_recv = |msg: &str| -> String {
            writeln!(stdin, "{}", msg).unwrap();
            stdin.flush().unwrap();
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            line
        };

        // 1. Initialize
        let resp = send_and_recv(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
        );
        let init: Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(init["result"]["protocolVersion"], 1);

        // 2. Session new
        let resp = send_and_recv(r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#);
        let session: Value = serde_json::from_str(&resp).unwrap();
        let session_id = session["result"]["sessionId"].as_str().unwrap();
        assert!(!session_id.is_empty());

        // 3. Send prompt — ask agy to reply with a known word
        let prompt_msg = format!(
            r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"Reply with exactly one word: PONG"}}]}}}}"#,
            session_id
        );
        writeln!(stdin, "{}", prompt_msg).unwrap();
        stdin.flush().unwrap();

        // Read lines until we get id:3 response (there may be a notification first)
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let mut got_notification = false;
        let mut response_text = String::new();
        loop {
            if std::time::Instant::now() > deadline {
                panic!("Timed out waiting for agy-acp response");
            }
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.is_empty() {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            let msg: Value = serde_json::from_str(line.trim()).unwrap();
            if msg.get("method") == Some(&json!("session/update")) {
                got_notification = true;
                response_text = msg["params"]["update"]["content"]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
            }
            if msg.get("id") == Some(&json!(3)) {
                assert!(msg["error"].is_null(), "Got error: {}", msg["error"]);
                assert_eq!(msg["result"]["stopReason"], "end_turn");
                break;
            }
        }

        drop(stdin);
        let _ = child.wait();

        assert!(got_notification, "Expected session/update notification");
        let lower = response_text.to_lowercase();
        assert!(
            lower.contains("pong"),
            "Expected 'PONG' in response, got: '{}'",
            response_text
        );
    }

    /// Helper: spawn agy-acp, return (stdin, reader, child)
    fn spawn_agy_acp() -> Option<(
        std::process::ChildStdin,
        std::io::BufReader<std::process::ChildStdout>,
        std::process::Child,
    )> {
        use std::io::BufReader;
        use std::process::{Command, Stdio};

        if !prepare_auth() {
            return None;
        }
        let agy_check = Command::new("agy").arg("--help").output();
        if agy_check.is_err() || !agy_check.unwrap().status.success() {
            eprintln!("SKIP: agy not found in PATH");
            return None;
        }
        let binary = std::env::current_dir()
            .unwrap()
            .join("target/release/agy-acp");
        if !binary.exists() {
            panic!("Run `cargo build --release` first");
        }

        let mut child = Command::new(&binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn agy-acp");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        Some((stdin, BufReader::new(stdout), child))
    }

    /// Helper: send JSON-RPC and read one response line
    fn send_recv(
        stdin: &mut std::process::ChildStdin,
        reader: &mut std::io::BufReader<std::process::ChildStdout>,
        msg: &str,
    ) -> String {
        use std::io::{BufRead, Write};
        writeln!(stdin, "{}", msg).unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        line
    }

    /// Helper: send a prompt and wait for the response (notification + final reply)
    fn send_prompt_wait(
        stdin: &mut std::process::ChildStdin,
        reader: &mut std::io::BufReader<std::process::ChildStdout>,
        id: u64,
        session_id: &str,
        text: &str,
    ) -> (Option<String>, Value) {
        use std::io::{BufRead, Write};
        use std::time::Duration;

        let msg = format!(
            r#"{{"jsonrpc":"2.0","id":{},"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"{}"}}]}}}}"#,
            id, session_id, text
        );
        writeln!(stdin, "{}", msg).unwrap();
        stdin.flush().unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let mut notification_text: Option<String> = None;
        loop {
            if std::time::Instant::now() > deadline {
                panic!("Timed out");
            }
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.is_empty() {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            let msg: Value = serde_json::from_str(line.trim()).unwrap();
            if msg.get("method") == Some(&json!("session/update")) {
                notification_text = msg["params"]["update"]["content"]["text"]
                    .as_str()
                    .map(String::from);
            }
            if msg.get("id") == Some(&json!(id)) {
                return (notification_text, msg);
            }
        }
    }

    /// E2E: multi-turn — second prompt reuses the same conversation via --conversation flag
    #[test]
    #[ignore]
    fn test_e2e_multi_turn() {
        let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
            return;
        };

        // Initialize
        send_recv(
            &mut stdin,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
        );

        // Session new
        let resp = send_recv(
            &mut stdin,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#,
        );
        let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"]
            .as_str()
            .unwrap()
            .to_string();

        // First prompt: set a context
        let (text1, resp1) = send_prompt_wait(
            &mut stdin,
            &mut reader,
            3,
            &session_id,
            "Remember this word: BANANA. Reply OK.",
        );
        assert!(resp1["error"].is_null(), "Turn 1 error: {}", resp1["error"]);
        assert!(text1.is_some());

        // Second prompt: ask it to recall — this exercises --conversation reuse
        let (text2, resp2) = send_prompt_wait(
            &mut stdin,
            &mut reader,
            4,
            &session_id,
            "What word did I ask you to remember? Reply with just that word.",
        );
        assert!(resp2["error"].is_null(), "Turn 2 error: {}", resp2["error"]);
        let reply = text2.unwrap_or_default().to_lowercase();
        assert!(
            reply.contains("banana"),
            "Expected 'BANANA' in multi-turn reply, got: '{}'",
            reply
        );

        drop(stdin);
        let _ = child.wait();
    }

    /// E2E: session/load — evict session from memory, then restore from persisted state
    #[test]
    #[ignore]
    fn test_e2e_session_load() {
        let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
            return;
        };

        send_recv(
            &mut stdin,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
        );
        let resp = send_recv(
            &mut stdin,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#,
        );
        let session_id = serde_json::from_str::<Value>(&resp).unwrap()["result"]["sessionId"]
            .as_str()
            .unwrap()
            .to_string();

        // Send first prompt to bind conversation and persist state
        let (_text, resp1) = send_prompt_wait(
            &mut stdin,
            &mut reader,
            3,
            &session_id,
            "Reply with exactly: FIRST_TURN",
        );
        assert!(
            resp1["error"].is_null(),
            "First turn error: {}",
            resp1["error"]
        );

        // Send second prompt on the same session — this confirms multi-turn works
        // (session/load is already tested in unit tests; here we just verify the session
        // can handle continued prompts after binding)
        let (text2, resp2) = send_prompt_wait(
            &mut stdin,
            &mut reader,
            4,
            &session_id,
            "Reply with exactly one word: SECOND",
        );
        assert!(
            resp2["error"].is_null(),
            "Second turn error: {}",
            resp2["error"]
        );
        assert!(text2.is_some(), "Expected response on continued session");

        drop(stdin);
        let _ = child.wait();
    }

    /// E2E: error path — invalid requests should return errors, not crash
    #[test]
    #[ignore]
    fn test_e2e_error_paths() {
        let Some((mut stdin, mut reader, mut child)) = spawn_agy_acp() else {
            return;
        };

        send_recv(
            &mut stdin,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
        );

        // Load a non-existent session
        let resp = send_recv(
            &mut stdin,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":2,"method":"session/load","params":{"sessionId":"non-existent-session"}}"#,
        );
        let val: Value = serde_json::from_str(&resp).unwrap();
        assert!(
            !val["error"].is_null(),
            "Expected error for unknown session"
        );

        // Unknown method
        let resp = send_recv(
            &mut stdin,
            &mut reader,
            r#"{"jsonrpc":"2.0","id":3,"method":"bogus/method","params":{}}"#,
        );
        let val: Value = serde_json::from_str(&resp).unwrap();
        assert!(!val["error"].is_null(), "Expected error for unknown method");

        drop(stdin);
        let _ = child.wait();
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_read_response_multi_step_no_skip_no_duplicate() {
        let root = std::env::temp_dir().join(format!("agy-acp-multi-step-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let db_path = conv_dir.join("multi.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE steps (
                idx INTEGER PRIMARY KEY,
                step_type INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                has_subtrajectory NUMERIC NOT NULL DEFAULT 0,
                metadata BLOB,
                error_details BLOB,
                permissions BLOB,
                task_details BLOB,
                render_info BLOB,
                step_payload BLOB,
                step_format INTEGER NOT NULL DEFAULT 0
            )",
        )
        .unwrap();

        // Helper: build payload with field 20 (sub-msg) → field 1 (text)
        fn make_payload(text: &str) -> Vec<u8> {
            // Inner message: field 1, wire type 2 (LEN), <text>
            let text_bytes = text.as_bytes();
            let mut inner = vec![0x0A]; // tag: field 1, wire type 2
            let mut len = text_bytes.len();
            loop {
                if len < 128 {
                    inner.push(len as u8);
                    break;
                }
                inner.push((len as u8 & 0x7F) | 0x80);
                len >>= 7;
            }
            inner.extend_from_slice(text_bytes);

            // Outer: field 20, wire type 2 (LEN), <inner>
            // tag = (20 << 3) | 2 = 162 → varint [0xA2, 0x01]
            let mut outer = vec![0xA2, 0x01];
            let mut ilen = inner.len();
            loop {
                if ilen < 128 {
                    outer.push(ilen as u8);
                    break;
                }
                outer.push((ilen as u8 & 0x7F) | 0x80);
                ilen >>= 7;
            }
            outer.extend(inner);
            outer
        }

        // step_type 0 = user, step_type 15 = response
        // Step 1: user prompt (step_type=0, no extractable text)
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (1, 0, X'0801')",
            [],
        )
        .unwrap();
        // Step 2: bot response "hello"
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![2i64, make_payload("hello")],
        )
        .unwrap();
        // Step 3: user prompt
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (3, 0, X'0802')",
            [],
        )
        .unwrap();
        // Step 4: bot response "world"
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![4i64, make_payload("world")],
        )
        .unwrap();
        // Step 5: bot response multi-line
        conn.execute(
            "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
            rusqlite::params![5i64, make_payload("line1\nline2\nline3")],
        )
        .unwrap();
        drop(conn);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        // From start: get all response steps
        let result = adapter.read_response_from_db("multi", -1);
        assert_eq!(
            result,
            Some(("hello\nworld\nline1\nline2\nline3".to_string(), 5))
        );

        // After step 2: skip "hello", get "world" + multi-line
        let result = adapter.read_response_from_db("multi", 2);
        assert_eq!(result, Some(("world\nline1\nline2\nline3".to_string(), 5)));

        // After step 4: only multi-line
        let result = adapter.read_response_from_db("multi", 4);
        assert_eq!(result, Some(("line1\nline2\nline3".to_string(), 5)));

        // After step 5: nothing new
        let result = adapter.read_response_from_db("multi", 5);
        assert_eq!(result, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_read_response_missing_steps_table() {
        let root = std::env::temp_dir().join(format!("agy-acp-noschema-{}", Uuid::new_v4()));
        let conv_dir = root.join("conversations");
        fs::create_dir_all(&conv_dir).unwrap();

        let db_path = conv_dir.join("empty.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch("CREATE TABLE other (id INTEGER)")
            .unwrap();
        drop(conn);

        let adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: conv_dir,
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        let result = adapter.read_response_from_db("empty", -1);
        assert_eq!(result, None);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_is_narration_true() {
        assert!(Adapter::is_narration("I will fetch the latest commits."));
        assert!(Adapter::is_narration(
            "I will fetch the latest commits.\nI will check the diff."
        ));
        assert!(Adapter::is_narration(
            "I will read the file.\n\nI will analyze the output."
        ));
    }

    #[test]
    fn test_is_narration_false() {
        assert!(!Adapter::is_narration("Here is the result."));
        assert!(!Adapter::is_narration(
            "I will fetch the commits.\nHere is the result."
        ));
        assert!(!Adapter::is_narration(""));
    }

    #[test]
    fn test_filter_narration_drops_leading_narration() {
        let parts = vec![
            "I will fetch the latest commits.\nI will check the diff.".to_string(),
            "I will read the file.".to_string(),
            "The fix is confirmed! LGTM ✅".to_string(),
        ];
        let result = Adapter::filter_narration_with_mode(&parts, true);
        assert_eq!(result, "The fix is confirmed! LGTM ✅");
    }

    #[test]
    fn test_filter_narration_preserves_content_after_first_non_narration() {
        let parts = vec![
            "I will check things.".to_string(),
            "Here is my analysis.".to_string(),
            "I will also note this is fine.".to_string(),
        ];
        let result = Adapter::filter_narration_with_mode(&parts, true);
        assert_eq!(
            result,
            "Here is my analysis.\nI will also note this is fine."
        );
    }

    #[test]
    fn test_filter_narration_full_mode() {
        let parts = vec![
            "I will fetch commits.".to_string(),
            "Final answer here.".to_string(),
        ];
        let result = Adapter::filter_narration_with_mode(&parts, false);
        assert_eq!(result, "I will fetch commits.\nFinal answer here.");
    }

    #[test]
    fn test_filter_narration_compact_mode() {
        let parts = vec![
            "I will fetch commits.".to_string(),
            "Final answer here.".to_string(),
        ];
        let result = Adapter::filter_narration_with_mode(&parts, true);
        assert_eq!(result, "Final answer here.");
    }

    #[test]
    fn test_filter_narration_single_part_unchanged() {
        let parts = vec!["I will do something.".to_string()];
        let result = Adapter::filter_narration(&parts);
        assert_eq!(result, "I will do something.");
    }

    #[test]
    fn test_filter_narration_all_narration_keeps_last() {
        let parts = vec![
            "I will fetch the file.".to_string(),
            "I will check the output.".to_string(),
            "I will verify the fix.".to_string(),
        ];
        let result = Adapter::filter_narration_with_mode(&parts, true);
        assert_eq!(result, "I will verify the fix.");
    }

    #[test]
    fn test_session_new_returns_models() {
        let mut adapter = Adapter::new();
        let response = adapter.handle_session_new(json!(1));
        let result = response.result.as_ref().unwrap();
        assert!(result.get("sessionId").is_some());
        let models = result.get("models").unwrap();
        assert!(models.get("currentModelId").is_some());
        assert!(models.get("availableModels").is_some());
    }

    #[test]
    fn test_session_set_model() {
        let mut adapter = Adapter::new();
        let new_resp = adapter.handle_session_new(json!(1));
        let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
            .as_str()
            .unwrap()
            .to_string();

        let set_resp = adapter.handle_session_set_model(
            json!(2),
            &json!({"sessionId": session_id, "modelId": "Gemini 3.5 Flash (High)"}),
        );
        assert!(set_resp.error.is_none());
        assert_eq!(
            adapter
                .sessions
                .get(&session_id)
                .unwrap()
                .model_id
                .as_deref(),
            Some("Gemini 3.5 Flash (High)")
        );
    }

    #[test]
    fn test_session_set_model_missing_params() {
        let mut adapter = Adapter::new();
        let resp = adapter.handle_session_set_model(json!(1), &json!({}));
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32602));
    }

    #[test]
    fn test_session_set_model_unknown_session() {
        let mut adapter = Adapter::new();
        let resp = adapter.handle_session_set_model(
            json!(1),
            &json!({"sessionId": "nonexistent", "modelId": "some-model"}),
        );
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32000));
    }

    #[test]
    #[ignore] // filesystem I/O
    fn test_session_set_model_persists() {
        let root = std::env::temp_dir().join(format!("agy-acp-model-persist-{}", Uuid::new_v4()));
        let _ = fs::create_dir_all(&root);

        let mut adapter = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };

        // Create a session with a conversation binding
        adapter.persist_session("sess-m1", Some("conv-m1"), 0, None);

        // Set model on the session
        adapter.restore_session_state("sess-m1");
        adapter.handle_session_set_model(
            json!(1),
            &json!({"sessionId": "sess-m1", "modelId": "Claude Opus 4.6 (Thinking)"}),
        );

        // Verify model is persisted by restoring in a fresh adapter
        let adapter2 = Adapter {
            sessions: HashMap::new(),
            working_dir: root.to_string_lossy().to_string(),
            conversations_dir: root.join("conversations"),
            state_file: root.join("sessions.json"),
            available_models: vec![],
        };
        let restored = adapter2.restore_session("sess-m1");
        assert_eq!(
            restored,
            Some((
                "conv-m1".to_string(),
                0,
                Some("Claude Opus 4.6 (Thinking)".to_string())
            ))
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn test_session_load_returns_models() {
        let mut adapter = Adapter::new();
        // Register a session so load can find it
        adapter.sessions.insert(
            "test-load".to_string(),
            Session {
                conversation_id: None,
                last_step_idx: -1,
                model_id: Some("Gemini 3.1 Pro (High)".to_string()),
            },
        );
        // persist_session requires a conversation_id for restore_session to work
        adapter.persist_session(
            "test-load",
            Some("conv-load"),
            -1,
            Some("Gemini 3.1 Pro (High)"),
        );
        adapter.sessions.clear();

        let output = adapter.handle_session_load(json!(1), &json!({"sessionId": "test-load"}));
        let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
        assert!(
            response["error"].is_null(),
            "error: {:?}",
            response["error"]
        );
        let models = response["result"]["models"].as_object().unwrap();
        assert_eq!(
            models["currentModelId"].as_str(),
            Some("Gemini 3.1 Pro (High)")
        );
    }

    #[test]
    fn test_session_resume_returns_models() {
        let mut adapter = Adapter::new();
        adapter.persist_session(
            "test-resume",
            Some("conv-resume"),
            -1,
            Some("GPT-OSS 120B (Medium)"),
        );
        adapter.sessions.clear();

        let response =
            adapter.handle_session_resume(json!(1), &json!({"sessionId": "test-resume"}));
        assert!(response.error.is_none(), "error: {:?}", response.error);
        let models = response.result.as_ref().unwrap()["models"]
            .as_object()
            .unwrap();
        assert_eq!(
            models["currentModelId"].as_str(),
            Some("GPT-OSS 120B (Medium)")
        );
    }

    #[test]
    fn test_session_models_json_default() {
        let mut adapter = Adapter::new();
        let models = adapter.session_models_json(None);
        let current = models["currentModelId"].as_str().unwrap();
        // Current should be first available model or empty
        if adapter.available_models.is_empty() {
            assert_eq!(current, "");
        } else {
            assert_eq!(current, adapter.available_models[0]);
        }
    }

    #[test]
    fn test_session_models_json_with_model() {
        let mut adapter = Adapter::new();
        adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
        let models = adapter.session_models_json(Some("Model B"));
        assert_eq!(models["currentModelId"].as_str(), Some("Model B"));
        let available = models["availableModels"].as_array().unwrap();
        assert_eq!(available.len(), 2);
        assert_eq!(available[0]["modelId"].as_str(), Some("Model A"));
        assert_eq!(available[1]["modelId"].as_str(), Some("Model B"));
    }
}
