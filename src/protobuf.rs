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

fn get_proto_fields(blob: &[u8], target: u64) -> Vec<Vec<u8>> {
    let mut i = 0;
    let mut fields = Vec::new();
    while i < blob.len() {
        let Some((tag, consumed)) = read_varint(&blob[i..]) else {
            return fields;
        };
        i += consumed;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;
        match wire_type {
            0 => {
                let start = i;
                let Some((_, c)) = read_varint(&blob[i..]) else {
                    return fields;
                };
                if field_number == target {
                    fields.push(blob[start..start + c].to_vec());
                }
                i += c;
            }
            2 => {
                let Some((len, c)) = read_varint(&blob[i..]) else {
                    return fields;
                };
                i += c;
                let len = len as usize;
                if i + len > blob.len() {
                    return fields;
                }
                if field_number == target {
                    fields.push(blob[i..i + len].to_vec());
                }
                i += len;
            }
            5 => {
                if i + 4 > blob.len() {
                    return fields;
                }
                if field_number == target {
                    fields.push(blob[i..i + 4].to_vec());
                }
                i += 4;
            }
            1 => {
                if i + 8 > blob.len() {
                    return fields;
                }
                if field_number == target {
                    fields.push(blob[i..i + 8].to_vec());
                }
                i += 8;
            }
            _ => return fields,
        }
    }
    fields
}

fn get_varint_field(blob: &[u8], target: u64) -> Option<u64> {
    let bytes = get_proto_fields(blob, target).into_iter().next()?;
    read_varint(&bytes).map(|(value, _)| value)
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
    } else if lower.contains("read") || lower.contains("view") || lower.contains("list") {
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
    matches!(step_type, 5 | 7 | 8 | 9 | 17 | 21 | 33 | 91 | 101 | 138)
}

/// Step type emitted by agy for the `generate_image` tool.
pub const GENERATE_IMAGE_STEP_TYPE: i64 = 91;

/// Image artifact extracted from a step_payload's field 104 (generate_image metadata).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageArtifact {
    pub absolute_path: String,
    pub mime: Option<String>,
    pub name: Option<String>,
}

const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "webp", "gif", "avif", "bmp", "tiff", "tif",
];

fn has_image_extension(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    IMAGE_EXTENSIONS
        .iter()
        .any(|ext| lower.ends_with(&format!(".{ext}")))
}

/// Strip a `file://` scheme and percent-decode the path. Returns the absolute path.
///
/// `file:///Users/x/a.png` → `/Users/x/a.png`
/// `file:///tmp/my%20image.png` → `/tmp/my image.png`
/// `file:///tmp/%E3%81%82.png` → `/tmp/あ.png`
/// `/Users/x/a.png` → `/Users/x/a.png` (bare paths are returned verbatim — they
/// are not URL-encoded, so a literal `%` must round-trip unchanged.)
fn normalize_file_uri(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (path, decode) = if let Some(rest) = trimmed.strip_prefix("file://") {
        // file:///abs/path → /abs/path; file://localhost/abs/path → /abs/path
        let stripped = rest.strip_prefix("localhost").unwrap_or(rest);
        (stripped.to_string(), true)
    } else {
        (trimmed.to_string(), false)
    };
    if !path.starts_with('/') {
        return None;
    }
    if decode {
        percent_decode(&path)
    } else {
        Some(path)
    }
}

/// Decode RFC 3986 percent-encoding into a UTF-8 string. Returns `None` if the
/// resulting bytes are not valid UTF-8 (e.g. corrupted input). `%` followed by
/// non-hex is preserved literally so partially-encoded strings degrade gracefully.
fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).ok()
}

/// Extract a generated image artifact from a step_payload, materializing any
/// inline `data:image/...;base64,...` URI into the per-user default cache dir.
/// See [`extract_image_artifact_from_step_payload_in_dir`] for an injectable cache
/// dir variant used by tests.
pub fn extract_image_artifact_from_step_payload(blob: &[u8]) -> Option<ImageArtifact> {
    extract_image_artifact_from_step_payload_in_dir(blob, &default_image_cache_dir())
}

/// `extract_image_artifact_from_step_payload` with an explicit cache dir for inline
/// `data:` URIs — lets tests scope I/O to `$TMPDIR`.
///
/// Protobuf shape:
/// top.104 (image metadata) -> field 6 (media ref) -> field 5 (file URI / absolute path / data URI).
/// MIME comes from media ref field 1; ImageName from top.104 field 4.
/// `file://` and bare absolute paths are returned as-is; `data:image/...;base64,...`
/// is materialized into `cache_dir`.
pub fn extract_image_artifact_from_step_payload_in_dir(
    blob: &[u8],
    cache_dir: &std::path::Path,
) -> Option<ImageArtifact> {
    let image_meta = get_proto_field(blob, 104)?;
    let media = get_proto_field(&image_meta, 6)?;
    let raw_uri = get_text_field(&media, 5)?;
    let absolute_path = if let Some(path) = normalize_file_uri(&raw_uri) {
        if !has_image_extension(&path) {
            return None;
        }
        path
    } else if raw_uri.starts_with("data:image/") {
        materialize_data_uri_image(&raw_uri, cache_dir)?
    } else {
        return None;
    };
    let mime = get_text_field(&media, 1).filter(|m| !m.is_empty());
    let name = get_text_field(&image_meta, 4).filter(|n| !n.trim().is_empty());
    Some(ImageArtifact {
        absolute_path,
        mime,
        name,
    })
}

/// Default cache dir for materialized inline images. Prefers `~/.openab/agy-acp/images`
/// (sibling of `sessions.json`); falls back to the OS temp dir if `HOME` is unset.
pub fn default_image_cache_dir() -> std::path::PathBuf {
    std::env::var("HOME")
        .ok()
        .map(|h| std::path::PathBuf::from(h).join(".openab/agy-acp/images"))
        .unwrap_or_else(|| std::env::temp_dir().join("agy-acp-images"))
}

/// Decode a `data:image/<ext>;base64,<payload>` URI, write its bytes to `dir`, and
/// return the resulting absolute path. Returns `None` for raw (non-base64) data URIs,
/// non-image MIME types, or decode failures.
pub fn materialize_data_uri_image(
    data_uri: &str,
    dir: &std::path::Path,
) -> Option<String> {
    let rest = data_uri.strip_prefix("data:")?;
    let (header, body) = rest.split_once(',')?;
    if !header.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
        return None;
    }
    let mime = header.split(';').next()?.to_ascii_lowercase();
    if !mime.starts_with("image/") {
        return None;
    }
    let ext = mime_to_extension(&mime)?;
    let bytes = decode_base64(body)?;
    if bytes.is_empty() {
        return None;
    }
    std::fs::create_dir_all(dir).ok()?;
    let name = format!("agy-{:016x}.{}", fnv1a_64(&bytes), ext);
    let path = dir.join(&name);
    if !path.exists() {
        std::fs::write(&path, &bytes).ok()?;
    }
    Some(path.to_string_lossy().to_string())
}

fn mime_to_extension(mime: &str) -> Option<&'static str> {
    match mime {
        "image/png" => Some("png"),
        "image/jpeg" | "image/jpg" => Some("jpg"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        "image/avif" => Some("avif"),
        "image/bmp" => Some("bmp"),
        "image/tiff" => Some("tiff"),
        _ => None,
    }
}

fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn decode_base64(input: &str) -> Option<Vec<u8>> {
    fn lookup(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' | b'-' => Some(62),
            b'/' | b'_' => Some(63),
            _ => None,
        }
    }
    let cleaned: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(cleaned.len() * 3 / 4);
    for chunk in cleaned.chunks(4) {
        if chunk.len() < 2 {
            return None;
        }
        let mut v: u32 = 0;
        for &b in chunk {
            v = (v << 6) | (lookup(b)? as u32);
        }
        v <<= (4 - chunk.len()) * 6;
        out.push(((v >> 16) & 0xFF) as u8);
        if chunk.len() > 2 {
            out.push(((v >> 8) & 0xFF) as u8);
        }
        if chunk.len() > 3 {
            out.push((v & 0xFF) as u8);
        }
    }
    Some(out)
}

/// Format the markdown image link Paseo will display inline in the assistant message.
/// The URL portion has `)` and `\` escaped so paths with parentheses or backslashes
/// don't prematurely terminate the link.
pub fn image_markdown_message(absolute_path: &str) -> String {
    format!("![Generated image]({})", escape_markdown_url(absolute_path))
}

fn escape_markdown_url(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for c in path.chars() {
        match c {
            '\\' => out.push_str(r"\\"),
            ')' => out.push_str(r"\)"),
            other => out.push(other),
        }
    }
    out
}

fn fenced_code_block(text: &str) -> String {
    let mut fence_len = 3;
    let mut run_len = 0;
    for ch in text.chars() {
        if ch == '`' {
            run_len += 1;
            fence_len = fence_len.max(run_len + 1);
        } else {
            run_len = 0;
        }
    }

    let fence = "`".repeat(fence_len);
    format!("{fence}\n{text}\n{fence}")
}

pub fn tool_content(input: &Value, code_block: bool) -> Option<Value> {
    if let Some(text) = format_structured_tool_output(input) {
        return Some(json!({
            "type": "content",
            "content": { "type": "text", "text": fenced_code_block(&text) },
        }));
    }

    for key in [
        "thought",
        "thinking",
        "reasoning",
        "analysis",
        "plan",
        "text",
        "result",
        "output",
        "textOutput",
        "content",
        "summary",
    ] {
        if let Some(text) = input.get(key).and_then(|v| v.as_str()) {
            if !text.trim().is_empty() {
                let text = if code_block {
                    fenced_code_block(text)
                } else {
                    text.to_string()
                };
                return Some(json!({
                    "type": "content",
                    "content": { "type": "text", "text": text },
                }));
            }
        }
    }
    None
}

fn format_structured_tool_output(input: &Value) -> Option<String> {
    match input.get("resultType").and_then(|v| v.as_str()) {
        Some("grepSearch") => format_grep_output(input),
        Some("listDirectory") => format_list_output(input),
        _ => None,
    }
}

fn format_grep_output(input: &Value) -> Option<String> {
    if input
        .get("textOutput")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
    {
        return None;
    }

    let Some(hits) = input.get("hits").and_then(|v| v.as_array()) else {
        return Some("No matches".to_string());
    };
    if hits.is_empty() {
        return Some("No matches".to_string());
    }

    let lines: Vec<String> = hits
        .iter()
        .filter_map(|hit| {
            hit.as_object().map(|fields| {
                let mut parts: Vec<String> = fields
                    .iter()
                    .filter_map(|(key, value)| {
                        value
                            .as_str()
                            .map(String::from)
                            .or_else(|| value.as_i64().map(|n| n.to_string()))
                            .map(|value| (key, value))
                    })
                    .filter(|(_, value)| !value.trim().is_empty())
                    .map(|(key, value)| format!("{key}: {value}"))
                    .collect();
                parts.sort();
                parts.join(" | ")
            })
        })
        .filter(|line| !line.is_empty())
        .collect();

    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn format_list_output(input: &Value) -> Option<String> {
    let Some(entries) = input.get("entries").and_then(|v| v.as_array()) else {
        return Some("(empty directory)".to_string());
    };
    if entries.is_empty() {
        return Some("(empty directory)".to_string());
    }

    let lines: Vec<String> = entries
        .iter()
        .filter_map(|entry| {
            let name = entry.get("name").and_then(|v| v.as_str())?;
            if name.trim().is_empty() {
                return None;
            }
            let suffix = if entry
                .get("isDirectory")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                "/"
            } else {
                ""
            };
            Some(format!("{name}{suffix}"))
        })
        .collect();

    (!lines.is_empty()).then(|| lines.join("\n"))
}

pub fn tool_locations(input: &Value) -> Vec<Value> {
    let mut locations = Vec::new();
    for key in [
        "AbsolutePath",
        "SearchPath",
        "path",
        "file",
        "FilePath",
        "fileUri",
        "dirUri",
        "cwdUri",
    ] {
        if let Some(path) = input.get(key).and_then(|v| v.as_str()) {
            let mut loc = json!({ "path": path });
            if let Some(line) = input
                .get("StartLine")
                .or_else(|| input.get("startLine"))
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

fn parse_tool_run(
    blob: &[u8],
) -> Option<(Option<String>, Option<String>, Option<Value>, Option<String>)> {
    get_varint_field(blob, 1)?;
    let tool = get_proto_field(blob, 5)?;
    let call = get_proto_field(&tool, 4);
    let call_id = call.as_ref().and_then(|call| get_text_field(call, 1));
    let name = call
        .as_ref()
        .and_then(|call| get_text_field(call, 2).or_else(|| get_text_field(call, 9)));
    let raw_input = call
        .as_ref()
        .and_then(|call| get_text_field(call, 3))
        .and_then(|s| {
            serde_json::from_str::<Value>(&s)
                .ok()
                .or_else(|| extract_first_json_object(&s))
        });
    let title = get_text_field(&tool, 30)
        .or_else(|| get_text_field(&tool, 31))
        .or_else(|| {
            raw_input
                .as_ref()
                .and_then(|v| v.get("toolSummary").or_else(|| v.get("toolAction")))
                .and_then(|v| v.as_str())
                .map(String::from)
        });
    if name.is_none() && raw_input.is_none() {
        return None;
    }
    Some((call_id, name, raw_input, title))
}

fn parse_search_hits(grep: &[u8]) -> Vec<Value> {
    get_proto_fields(grep, 4)
        .into_iter()
        .map(|hit| {
            let mut out = json!({});
            for field in [1, 2, 3, 4, 5] {
                if let Some(text) = get_text_field(&hit, field) {
                    out[format!("field{field}")] = json!(text);
                } else if let Some(value) = get_varint_field(&hit, field) {
                    out[format!("field{field}")] = json!(value);
                }
            }
            out
        })
        .filter(|hit| hit.as_object().map(|o| !o.is_empty()).unwrap_or(false))
        .collect()
}

fn parse_tool_result(blob: &[u8]) -> Option<Value> {
    if let Some(write) = get_proto_field(blob, 10) {
        let mut out = json!({ "resultType": "writeFile" });
        if let Some(summary) = get_text_field(&write, 26) {
            out["summary"] = json!(summary);
        }
        return Some(out);
    }

    if let Some(grep) = get_proto_field(blob, 13) {
        let mut out = json!({ "resultType": "grepSearch" });
        for (key, field) in [
            ("query", 1),
            ("includeGlob", 2),
            ("textOutput", 3),
            ("shellCommand", 10),
            ("cwdUri", 11),
        ] {
            if let Some(text) = get_text_field(&grep, field) {
                out[key] = json!(text);
            }
        }
        let hits = parse_search_hits(&grep);
        if !hits.is_empty() {
            out["hits"] = Value::Array(hits);
        }
        return Some(out);
    }

    if let Some(view) = get_proto_field(blob, 14) {
        let mut out = json!({ "resultType": "viewFile" });
        if let Some(file_uri) = get_text_field(&view, 1) {
            out["fileUri"] = json!(file_uri);
        }
        for (key, field) in [
            ("startLine", 2),
            ("endLine", 3),
            ("nextLine", 11),
            ("fileSizeOrTotal", 12),
        ] {
            if let Some(value) = get_varint_field(&view, field) {
                out[key] = json!(value);
            }
        }
        if let Some(content) = get_text_field(&view, 4) {
            out["content"] = json!(content);
        }
        return Some(out);
    }

    if let Some(list) = get_proto_field(blob, 15) {
        let mut out = json!({ "resultType": "listDirectory" });
        if let Some(dir_uri) = get_text_field(&list, 1) {
            out["dirUri"] = json!(dir_uri);
        }
        let entries: Vec<Value> = get_proto_fields(&list, 3)
            .into_iter()
            .map(|entry| {
                json!({
                    "name": get_text_field(&entry, 1).unwrap_or_default(),
                    "isDirectory": get_varint_field(&entry, 2).unwrap_or(0) != 0,
                    "fileSize": get_varint_field(&entry, 4).unwrap_or(0),
                })
            })
            .filter(|entry| entry["name"].as_str().map(|s| !s.is_empty()).unwrap_or(false))
            .collect();
        if !entries.is_empty() {
            out["entries"] = Value::Array(entries);
        }
        return Some(out);
    }

    None
}

pub fn extract_tool_update_from_step_payload(
    idx: i64,
    step_type: i64,
    blob: &[u8],
) -> Option<Value> {
    let parsed_tool = parse_tool_run(blob);
    let parsed_result = parsed_tool.as_ref().and_then(|_| parse_tool_result(blob));
    let strings = extract_printable_strings(blob);
    let scraped_input = strings.iter().find_map(|s| {
        let trimmed = s.trim();
        if trimmed.starts_with('{') && trimmed.ends_with('}') {
            serde_json::from_str::<Value>(trimmed).ok()
        } else {
            extract_first_json_object(trimmed)
        }
    });

    let scraped_name = strings
        .iter()
        .find_map(|s| {
            let trimmed = s.trim();
            looks_like_tool_name(trimmed).then(|| trimmed.to_string())
        })
        .or_else(|| strings.iter().find_map(|s| extract_tool_name(s)));
    let (parsed_call_id, parsed_name, parsed_input, parsed_title) =
        parsed_tool.unwrap_or((None, None, None, None));
    let name = parsed_name.or(scraped_name);
    let raw_input = parsed_input.or(scraped_input);
    let raw_output = parsed_result;
    let title_from_input = raw_input
        .as_ref()
        .and_then(|v| v.get("toolSummary").or_else(|| v.get("toolAction")))
        .and_then(|v| v.as_str())
        .map(String::from);
    let fallback_kind = name.as_deref().map(tool_kind).unwrap_or("other");
    if parsed_title.is_none() && title_from_input.is_none() && fallback_kind == "other" {
        return None;
    }
    let title = parsed_title.or(title_from_input).or_else(|| name.clone())?;
    let name_kind = name.as_deref().map(tool_kind).unwrap_or("other");
    let title_kind = tool_kind(&title);
    let kind = if name_kind == "other" {
        title_kind
    } else {
        name_kind
    };
    let tool_call_id = parsed_call_id.unwrap_or_else(|| format!("agy-{idx}-{step_type}"));
    let mut locations = raw_input.as_ref().map(tool_locations).unwrap_or_default();
    if let Some(output_locations) = raw_output.as_ref().map(tool_locations) {
        locations.extend(output_locations);
    }
    let content = raw_output
        .as_ref()
        .and_then(|output| tool_content(output, true))
        .or_else(|| raw_input.as_ref().and_then(|input| tool_content(input, false)));

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
    if let Some(output) = raw_output {
        update["rawOutput"] = output;
    }
    if !locations.is_empty() {
        update["locations"] = Value::Array(locations);
    }
    if let Some(content) = content {
        update["content"] = Value::Array(vec![content]);
    }
    Some(update)
}
