use serde_json::{json, Value};

/// Read a protobuf varint, returning (value, bytes_consumed).
pub fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
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

/// Extract the first length-delimited field with the given number from a protobuf blob.
pub fn get_proto_field(blob: &[u8], target: u64) -> Option<Vec<u8>> {
    let mut i = 0;
    while i < blob.len() {
        let (tag, consumed) = read_varint(&blob[i..])?;
        i += consumed;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;
        match wire_type {
            0 => {
                let (_, c) = read_varint(&blob[i..])?;
                i += c;
            }
            2 => {
                let (len, c) = read_varint(&blob[i..])?;
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

pub fn get_text_field(blob: &[u8], target: u64) -> Option<String> {
    let bytes = get_proto_field(blob, target)?;
    String::from_utf8(bytes).ok()
}

/// Extract text from a step_payload protobuf: top-level field 20 (sub-message) → field 1 (string).
pub fn extract_text_from_step_payload(blob: &[u8]) -> Option<String> {
    let field_20 = get_proto_field(blob, 20)?;
    let field_1 = get_proto_field(&field_20, 1)?;
    String::from_utf8(field_1).ok()
}

pub fn extract_user_text_from_step_payload(blob: &[u8]) -> Option<String> {
    let prompt = get_proto_field(blob, 19)?;
    get_text_field(&prompt, 2)
        .or_else(|| {
            let content = get_proto_field(&prompt, 3)?;
            get_text_field(&content, 1)
        })
        .filter(|text| !text.trim().is_empty())
}

/// Extract a generated conversation title from a step type 23 payload:
/// top-level field 30 (sub-message) -> field 4 (string).
pub fn extract_title_from_step_payload(blob: &[u8]) -> Option<String> {
    let title_update = get_proto_field(blob, 30)?;
    get_text_field(&title_update, 4).filter(|title| !title.trim().is_empty())
}

pub fn extract_first_json_object(s: &str) -> Option<Value> {
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

pub fn extract_printable_strings(blob: &[u8]) -> Vec<String> {
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
    s.len() >= 2
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && (s.contains('_') || s.chars().any(|c| c.is_ascii_uppercase()))
        && s.len() <= 64
}

pub fn extract_tool_name(s: &str) -> Option<String> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .find(|part| looks_like_tool_name(part) && !part.starts_with("tool"))
        .map(String::from)
}

pub fn tool_kind(tool_name: &str) -> &'static str {
    let lower = tool_name.to_lowercase();
    if lower.contains("write") || lower.contains("edit") || lower.contains("patch") {
        "edit"
    } else if lower.contains("delete") || lower.contains("remove") {
        "delete"
    } else if lower.contains("move") || lower.contains("rename") {
        "move"
    } else if lower.contains("read") || lower.contains("view") {
        "read"
    } else if lower.contains("grep") || lower.contains("search") || lower.contains("find") {
        "search"
    } else if lower.contains("command") || lower.contains("execute") || lower.contains("terminal") {
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

pub fn is_tool_step_type(step_type: i64) -> bool {
    matches!(step_type, 5 | 7 | 8 | 9 | 17 | 21 | 33 | 101 | 138)
}

pub fn tool_content(input: &Value) -> Option<Value> {
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

pub fn tool_locations(input: &Value) -> Vec<Value> {
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

pub fn message_chunk_update(session_update: &str, text: String) -> Value {
    json!({
        "sessionUpdate": session_update,
        "content": { "type": "text", "text": text },
    })
}

pub fn extract_tool_update_from_step_payload(
    idx: i64,
    step_type: i64,
    blob: &[u8],
) -> Option<Value> {
    let strings = extract_printable_strings(blob);
    let raw_input = strings.iter().find_map(|s| {
        let trimmed = s.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            serde_json::from_str::<Value>(trimmed).ok()
        } else {
            extract_first_json_object(trimmed)
        }
    });

    let name = strings
        .iter()
        .find_map(|s| {
            let trimmed = s.trim();
            looks_like_tool_name(trimmed).then(|| trimmed.to_string())
        })
        .or_else(|| strings.iter().find_map(|s| extract_tool_name(s)));
    let title_from_input = raw_input
        .as_ref()
        .and_then(|v| v.get("toolSummary").or_else(|| v.get("toolAction")))
        .and_then(|v| v.as_str())
        .map(String::from);
    let fallback_kind = name.as_deref().map(tool_kind).unwrap_or("other");
    if title_from_input.is_none() && fallback_kind == "other" {
        return None;
    }
    let title = title_from_input.or_else(|| name.clone())?;
    let name_kind = name.as_deref().map(tool_kind).unwrap_or("other");
    let title_kind = tool_kind(&title);
    let kind = if title_kind == "other" {
        name_kind
    } else {
        title_kind
    };
    let tool_call_id = format!("agy-{idx}-{step_type}");
    let locations = raw_input.as_ref().map(tool_locations).unwrap_or_default();
    let content = raw_input.as_ref().and_then(tool_content);

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
