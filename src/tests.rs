use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

use crate::adapter::{filter_narration, Adapter};
use crate::protobuf::{
    extract_image_artifact_from_step_payload, extract_image_artifact_from_step_payload_in_dir,
    extract_text_from_step_payload, extract_title_from_step_payload, extract_tool_name,
    extract_tool_update_from_step_payload, extract_user_text_from_step_payload,
    image_markdown_message, is_tool_step_type, materialize_data_uri_image, read_varint,
};
use crate::streaming::poll_streaming_delta;
use crate::types::StreamingState;
use std::sync::{Arc, Mutex};
use crate::Cli;
use clap::Parser;

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

fn push_varint_field(out: &mut Vec<u8>, field_number: u64, value: u64) {
    push_varint(out, field_number << 3);
    push_varint(out, value);
}

fn make_assistant_payload(text: &str) -> Vec<u8> {
    let mut inner = Vec::new();
    push_len_field(&mut inner, 1, text.as_bytes());

    let mut outer = Vec::new();
    push_len_field(&mut outer, 20, &inner);
    outer
}

#[test]
fn test_parse_skip_naration_flag() {
    assert!(
        Cli::try_parse_from(["agy-acp", "--skip-naration"])
            .unwrap()
            .skip_naration
    );
    assert!(!Cli::try_parse_from(["agy-acp"]).unwrap().skip_naration);
    assert!(Cli::try_parse_from(["agy-acp", "--skip-narration"]).is_err());
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

fn make_title_payload(title: &str) -> Vec<u8> {
    let mut title_update = Vec::new();
    push_len_field(&mut title_update, 4, title.as_bytes());

    let mut outer = Vec::new();
    push_len_field(&mut outer, 30, &title_update);
    outer
}

fn make_tool_payload(
    call_id: &str,
    tool_name: &str,
    input_json: &str,
    summary: &str,
    result_field: Option<(u64, Vec<u8>)>,
) -> Vec<u8> {
    let mut call = Vec::new();
    push_len_field(&mut call, 1, call_id.as_bytes());
    push_len_field(&mut call, 2, tool_name.as_bytes());
    push_len_field(&mut call, 3, input_json.as_bytes());
    push_len_field(&mut call, 9, tool_name.as_bytes());

    let mut tool = Vec::new();
    push_len_field(&mut tool, 4, &call);
    push_len_field(&mut tool, 30, summary.as_bytes());

    let mut outer = Vec::new();
    push_varint_field(&mut outer, 1, 7);
    push_len_field(&mut outer, 5, &tool);
    if let Some((field, result)) = result_field {
        push_len_field(&mut outer, field, &result);
    }
    outer
}

#[test]
fn test_extract_text_from_step_payload_field20_field1() {
    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(0x05);
    inner.extend_from_slice(b"hello");

    let mut blob = vec![0x08, 0x0F, 0xA2, 0x01, inner.len() as u8];
    blob.extend_from_slice(&inner);
    assert_eq!(
        extract_text_from_step_payload(&blob),
        Some("hello".to_string())
    );
}

#[test]
fn test_extract_text_returns_none_without_field20() {
    let blob = vec![0x08, 0x03];
    assert_eq!(extract_text_from_step_payload(&blob), None);
}

#[test]
fn test_extract_user_text_from_step_payload_field19_field2() {
    let payload = make_user_payload("how are you?");
    assert_eq!(
        extract_user_text_from_step_payload(&payload),
        Some("how are you?".to_string())
    );
}

#[test]
fn test_extract_title_from_step_payload_field30_field4() {
    let payload = make_title_payload("Documenting Conversation Snapshot Function");
    assert_eq!(
        extract_title_from_step_payload(&payload),
        Some("Documenting Conversation Snapshot Function".to_string())
    );
}

#[test]
fn test_extract_title_ignores_empty_title() {
    assert_eq!(
        extract_title_from_step_payload(&make_title_payload("  ")),
        None
    );
}

#[test]
fn test_extract_text_multiline() {
    let text = b"Safe memory rules\nCompiler points out the flaws\nFast and fearless code";
    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(text.len() as u8);
    inner.extend_from_slice(text);

    let mut blob = vec![0x08, 0x01, 0xA2, 0x01, inner.len() as u8];
    blob.extend_from_slice(&inner);
    assert_eq!(
        extract_text_from_step_payload(&blob),
        Some(
            "Safe memory rules\nCompiler points out the flaws\nFast and fearless code".to_string()
        )
    );
}

#[test]
fn test_extract_tool_update_from_step_payload_json() {
    let payload = br#"
        grep_search
        {"Query":"prompt","SearchPath":"/tmp/project/src/main.rs","toolAction":"Finding prompt handling","toolSummary":"Grep prompt"}
    "#;

    let update = extract_tool_update_from_step_payload(19, 7, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "agy-19-7");
    assert_eq!(update["title"], "Grep prompt");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["status"], "completed");
    assert_eq!(update["rawInput"]["Query"], "prompt");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src/main.rs");
}

#[test]
/// Tests that when a tool payload lacks a JSON body (and thus has no `toolSummary`
/// or `toolAction`), the extractor falls back to using the extracted tool name
/// (e.g., `view_file`) as the update title.
fn test_extract_tool_update_uses_tool_name_fallback() {
    let payload = b"view_file";
    let update = extract_tool_update_from_step_payload(3, 8, payload).unwrap();
    assert_eq!(update["title"], "view_file");
    assert_eq!(update["kind"], "read");
}

#[test]
fn test_extract_tool_update_ignores_single_letter_noise() {
    let payload = b"P";
    assert_eq!(extract_tool_update_from_step_payload(4, 17, payload), None);
}

#[test]
fn test_extract_tool_update_ignores_generic_message_fallback() {
    let payload = b"Message";
    assert_eq!(extract_tool_update_from_step_payload(5, 17, payload), None);
}

#[test]
fn test_extract_tool_update_parses_first_balanced_json_object() {
    let payload = br#"
        abc123 view_file
        {"AbsolutePath":"/tmp/project/README.md","toolAction":"Reading README.md","toolSummary":"View README file"}
        trailing render blob {not json}
    "#;

    let update = extract_tool_update_from_step_payload(6, 8, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["title"], "View README file");
    assert_eq!(update["kind"], "read");
    assert_eq!(update["rawInput"]["AbsolutePath"], "/tmp/project/README.md");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/README.md");
}

#[test]
fn test_extract_tool_update_kind_prefers_tool_name_over_title() {
    let payload = br#"
        view_file
        {"AbsolutePath":"/tmp/project/flow_graph_write_node.go","toolSummary":"View flow_graph_write_node.go"}
    "#;

    let update = extract_tool_update_from_step_payload(7, 8, payload).unwrap();
    assert_eq!(update["title"], "View flow_graph_write_node.go");
    assert_eq!(update["kind"], "read");
}

#[test]
fn test_extract_tool_name_from_embedded_token() {
    assert_eq!(
        extract_tool_name("abc123\tview_file\n{...}"),
        Some("view_file".to_string())
    );
}

#[test]
fn test_extract_tool_update_from_pascal_case_edit_tool() {
    let payload = br#"
        Edit
        {"file_path":"/tmp/project/src/main.rs","old_string":"old","new_string":"new"}
    "#;

    let update = extract_tool_update_from_step_payload(9, 4, payload).unwrap();
    assert_eq!(update["title"], "Edit");
    assert_eq!(update["kind"], "edit");
    assert_eq!(update["rawInput"]["file_path"], "/tmp/project/src/main.rs");
}

#[test]
fn test_extract_tool_update_from_bash_tool() {
    let payload = br#"
        run_command
        {"CommandLine":"cargo test","Cwd":"/tmp/project","toolAction":"Running tests","toolSummary":"Run cargo test"}
    "#;

    let update = extract_tool_update_from_step_payload(10, 21, payload).unwrap();
    assert_eq!(update["title"], "Run cargo test");
    assert_eq!(update["kind"], "execute");
    assert_eq!(update["rawInput"]["CommandLine"], "cargo test");
}

#[test]
fn test_extract_tool_update_from_web_search_step() {
    let payload = br#"
        search_web
        {"query":"FIFA World Cup 2026 dates","toolAction":"Searching World Cup dates","toolSummary":"Search FIFA World Cup 2026 dates"}
    "#;

    assert!(is_tool_step_type(33));
    let update = extract_tool_update_from_step_payload(3, 33, payload).unwrap();
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "agy-3-33");
    assert_eq!(update["title"], "Search FIFA World Cup 2026 dates");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["status"], "completed");
    assert_eq!(update["rawInput"]["query"], "FIFA World Cup 2026 dates");
}

#[test]
fn test_extract_tool_update_maps_reasoning_to_think_content() {
    let payload = br#"
        thinking
        {"thought":"Need to inspect the protocol before changing serialization.","toolSummary":"Reasoning"}
    "#;

    let update = extract_tool_update_from_step_payload(21, 17, payload).unwrap();
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
fn test_extract_tool_update_from_structured_grep_payload() {
    let mut grep = Vec::new();
    push_len_field(&mut grep, 1, b"StepPayload");
    push_len_field(&mut grep, 2, b"src/*.rs");
    push_len_field(&mut grep, 3, b"src/protobuf.rs:1:message StepPayload");
    push_len_field(&mut grep, 10, b"rg StepPayload src");
    push_len_field(&mut grep, 11, b"file:///tmp/project");
    let payload = make_tool_payload(
        "0t0p5kn3",
        "grep_search",
        r#"{"SearchPath":"/tmp/project/src","toolAction":"Searching protobuf schema"}"#,
        "Proto search",
        Some((13, grep)),
    );

    let update = extract_tool_update_from_step_payload(22, 7, &payload).unwrap();
    assert_eq!(update["toolCallId"], "0t0p5kn3");
    assert_eq!(update["title"], "Proto search");
    assert_eq!(update["kind"], "search");
    assert_eq!(update["rawInput"]["SearchPath"], "/tmp/project/src");
    assert_eq!(update["rawOutput"]["query"], "StepPayload");
    assert_eq!(
        update["rawOutput"]["textOutput"],
        "src/protobuf.rs:1:message StepPayload"
    );
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src");
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nsrc/protobuf.rs:1:message StepPayload\n```"
    );
}

#[test]
fn test_extract_tool_update_formats_structured_grep_hits_without_text_output() {
    let mut hit = Vec::new();
    push_len_field(&mut hit, 1, b"src/protobuf.rs");
    push_varint_field(&mut hit, 2, 42);
    push_len_field(&mut hit, 3, b"fn parse_tool_result(blob: &[u8]) -> Option<Value> {");

    let mut grep = Vec::new();
    push_len_field(&mut grep, 1, b"parse_tool_result");
    push_len_field(&mut grep, 4, &hit);
    let payload = make_tool_payload(
        "grep-hit-call",
        "grep_search",
        r#"{"SearchPath":"/tmp/project/src","toolAction":"Searching parser"}"#,
        "Parser search",
        Some((13, grep)),
    );

    let update = extract_tool_update_from_step_payload(26, 7, &payload).unwrap();
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nfield1: src/protobuf.rs | field2: 42 | field3: fn parse_tool_result(blob: &[u8]) -> Option<Value> {\n```"
    );
}

#[test]
fn test_extract_tool_update_from_structured_view_payload() {
    let mut view = Vec::new();
    push_len_field(&mut view, 1, b"file:///tmp/project/src/protobuf.rs");
    push_varint_field(&mut view, 2, 10);
    push_varint_field(&mut view, 3, 12);
    push_len_field(&mut view, 4, b"pub fn read_varint() {}\n```");
    push_varint_field(&mut view, 11, 13);
    push_varint_field(&mut view, 12, 200);
    let payload = make_tool_payload("view-call", "view_file", "{}", "Viewing file", Some((14, view)));

    let update = extract_tool_update_from_step_payload(23, 8, &payload).unwrap();
    assert_eq!(update["title"], "Viewing file");
    assert_eq!(update["kind"], "read");
    assert_eq!(update["rawOutput"]["fileUri"], "file:///tmp/project/src/protobuf.rs");
    assert_eq!(update["rawOutput"]["startLine"], 10);
    assert_eq!(update["locations"][0]["path"], "file:///tmp/project/src/protobuf.rs");
    assert_eq!(update["locations"][0]["line"], 10);
    assert_eq!(
        update["content"][0]["content"]["text"],
        "````\npub fn read_varint() {}\n```\n````"
    );
}

#[test]
fn test_extract_tool_update_from_structured_list_payload() {
    let mut entry = Vec::new();
    push_len_field(&mut entry, 1, b"src");
    push_varint_field(&mut entry, 2, 1);
    push_varint_field(&mut entry, 4, 0);

    let mut list = Vec::new();
    push_len_field(&mut list, 1, b"file:///tmp/project");
    push_len_field(&mut list, 3, &entry);
    let payload = make_tool_payload("list-call", "list_dir", "{}", "Listing directory", Some((15, list)));

    let update = extract_tool_update_from_step_payload(24, 9, &payload).unwrap();
    assert_eq!(update["title"], "Listing directory");
    assert_eq!(update["kind"], "read");
    assert_eq!(update["rawOutput"]["dirUri"], "file:///tmp/project");
    assert_eq!(update["rawOutput"]["entries"][0]["name"], "src");
    assert_eq!(update["rawOutput"]["entries"][0]["isDirectory"], true);
    assert_eq!(update["content"][0]["content"]["text"], "```\nsrc/\n```");
}

#[test]
fn test_extract_tool_update_formats_empty_structured_list_payload() {
    let mut list = Vec::new();
    push_len_field(&mut list, 1, b"file:///tmp/project");
    let payload = make_tool_payload(
        "empty-list-call",
        "list_dir",
        "{}",
        "Listing directory",
        Some((15, list)),
    );

    let update = extract_tool_update_from_step_payload(27, 9, &payload).unwrap();
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\n(empty directory)\n```"
    );
}

#[test]
fn test_extract_tool_update_from_structured_write_payload() {
    let mut write = Vec::new();
    push_len_field(&mut write, 26, b"Wrote 42 bytes");
    let payload = make_tool_payload(
        "write-call",
        "write_to_file",
        r#"{"AbsolutePath":"/tmp/project/src/main.rs"}"#,
        "Writing file",
        Some((10, write)),
    );

    let update = extract_tool_update_from_step_payload(25, 5, &payload).unwrap();
    assert_eq!(update["title"], "Writing file");
    assert_eq!(update["kind"], "edit");
    assert_eq!(update["rawOutput"]["summary"], "Wrote 42 bytes");
    assert_eq!(update["locations"][0]["path"], "/tmp/project/src/main.rs");
    assert_eq!(
        update["content"][0]["content"]["text"],
        "```\nWrote 42 bytes\n```"
    );
}

#[test]
fn test_read_varint() {
    assert_eq!(read_varint(&[0x05]), Some((5, 1)));
    assert_eq!(read_varint(&[0xAC, 0x02]), Some((300, 2)));
    assert_eq!(read_varint(&[]), None);
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
#[ignore]
fn test_session_load_restores_persisted_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-load-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
    };
    adapter.persist_session("sess-1", Some("conv-abc"), 5, None);

    let output = adapter.handle_session_load(json!(7), &json!({"sessionId": "sess-1"}));
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
#[ignore]
fn test_session_load_rejects_unknown_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-missing-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
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
#[ignore]
fn test_session_load_replays_conversation_history() {
    let root = std::env::temp_dir().join(format!("agy-acp-load-replay-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

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
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 5, ?2)",
        rusqlite::params![
            4i64,
            br#"replace_file_content
            {"AbsolutePath":"/tmp/project/README.md","toolAction":"Editing README.md","toolSummary":"Edit README file"}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 21, ?2)",
        rusqlite::params![
            5i64,
            br#"run_command
            {"CommandLine":"cargo test","Cwd":"/tmp/project","toolAction":"Running tests","toolSummary":"Run cargo test"}"#
                .as_slice()
        ],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![6i64, make_assistant_payload("hello from agent")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 14, ?2)",
        rusqlite::params![7i64, make_user_payload("how are you?")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![8i64, make_assistant_payload("second response")],
    )
    .unwrap();
    drop(conn);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: conv_dir,
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
    };
    adapter.persist_session("sess-replay", Some("conv-replay"), 8, None);

    let output = adapter.handle_session_load(json!(1), &json!({"sessionId": "sess-replay"}));

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
    assert!(updates.iter().any(|notification| {
        notification["params"]["update"]["title"] == "Edit README file"
            && notification["params"]["update"]["kind"] == "edit"
    }));
    assert!(updates.iter().any(|notification| {
        notification["params"]["update"]["title"] == "Run cargo test"
            && notification["params"]["update"]["kind"] == "execute"
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
            "tool_call",
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

    let response: Value = serde_json::from_str(output.last().unwrap()).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(
        response["result"]["sessionId"].as_str(),
        Some("sess-replay")
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_resume_restores_persisted_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
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
#[ignore]
fn test_session_resume_rejects_unknown_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-miss-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
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
        skip_naration: false,
    };
    adapter.sessions.insert(
        "sess-memory".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
        },
    );

    let response = adapter.handle_session_resume(json!(12), &json!({"sessionId": "sess-memory"}));

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
        skip_naration: false,
    };
    adapter.sessions.insert(
        "sess-memory-load".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: None,
        },
    );

    let output = adapter.handle_session_load(json!(13), &json!({"sessionId": "sess-memory-load"}));

    assert_eq!(output.len(), 1);
    let response: Value = serde_json::from_str(&output[0]).unwrap();
    assert!(response["error"].is_null());
    assert_eq!(response["result"]["sessionId"], "sess-memory-load");
}

#[test]
#[ignore]
fn test_session_resume_does_not_replay_history() {
    let root = std::env::temp_dir().join(format!("agy-acp-resume-noreplay-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
    };
    adapter.persist_session("sess-nr", Some("conv-nr"), 10, None);

    let response = adapter.handle_session_resume(json!(13), &json!({"sessionId": "sess-nr"}));
    assert!(response.error.is_none());
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
#[ignore]
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
        skip_naration: false,
    };

    let before = adapter.conversation_snapshot();
    assert!(before.contains("existing"));

    fs::write(conv_dir.join("new-conv.db"), b"new").unwrap();
    fs::write(conv_dir.join("new-conv.db-wal"), b"wal").unwrap();
    fs::write(conv_dir.join("new-conv.db-shm"), b"shm").unwrap();

    assert_eq!(
        adapter.new_conversation_id(&before),
        Some("new-conv".to_string())
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
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
        skip_naration: false,
    };

    let before = adapter.conversation_snapshot();
    fs::write(conv_dir.join("a.db"), b"").unwrap();
    fs::write(conv_dir.join("b.db"), b"").unwrap();

    assert_eq!(adapter.new_conversation_id(&before), None);
    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_persist_and_restore_session() {
    let root = std::env::temp_dir().join(format!("agy-acp-state-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
    };

    adapter.persist_session("sess-1", Some("conv-abc"), 7, None);
    let restored = adapter.restore_session("sess-1");
    assert_eq!(restored, Some(("conv-abc".to_string(), 7, None)));

    let missing = adapter.restore_session("sess-unknown");
    assert_eq!(missing, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_read_response_from_db() {
    let root = std::env::temp_dir().join(format!("agy-acp-sqlite-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

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

    let mut inner = Vec::new();
    inner.push(0x0A);
    inner.push(11);
    inner.extend_from_slice(b"hello world");
    let mut payload = vec![0x08, 0x0F, 0xA2, 0x01, inner.len() as u8];
    payload.extend_from_slice(&inner);

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![1i64, payload],
    )
    .unwrap();

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
        skip_naration: false,
    };

    let result = adapter.read_response_from_db("test-conv", -1);
    assert_eq!(result, Some(("hello world".to_string(), 1)));

    let result = adapter.read_response_from_db("test-conv", 1);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

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

#[test]
#[ignore]
fn test_e2e_agy_acp_full_round_trip() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    use std::time::Duration;

    if !prepare_auth() {
        return;
    }

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

    let mut send_and_recv = |msg: &str| -> String {
        writeln!(stdin, "{}", msg).unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        line
    };

    let resp = send_and_recv(
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientName":"e2e","clientVersion":"0.1"}}"#,
    );
    let init: Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(init["result"]["protocolVersion"], 1);

    let resp = send_and_recv(r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#);
    let session: Value = serde_json::from_str(&resp).unwrap();
    let session_id = session["result"]["sessionId"].as_str().unwrap();
    assert!(!session_id.is_empty());

    let prompt_msg = format!(
        r#"{{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{{"sessionId":"{}","prompt":[{{"type":"text","text":"Reply with exactly one word: PONG"}}]}}}}"#,
        session_id
    );
    writeln!(stdin, "{}", prompt_msg).unwrap();
    stdin.flush().unwrap();

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

#[test]
#[ignore]
fn test_e2e_multi_turn() {
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

    let (text1, resp1) = send_prompt_wait(
        &mut stdin,
        &mut reader,
        3,
        &session_id,
        "Remember this word: BANANA. Reply OK.",
    );
    assert!(resp1["error"].is_null(), "Turn 1 error: {}", resp1["error"]);
    assert!(text1.is_some());

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
#[ignore]
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

    fn make_payload(text: &str) -> Vec<u8> {
        let text_bytes = text.as_bytes();
        let mut inner = vec![0x0A];
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

    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (1, 0, X'0801')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![2i64, make_payload("hello")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (3, 0, X'0802')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![4i64, make_payload("world")],
    )
    .unwrap();
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
        skip_naration: false,
    };

    let result = adapter.read_response_from_db("multi", -1);
    assert_eq!(
        result,
        Some(("hello\nworld\nline1\nline2\nline3".to_string(), 5))
    );

    let result = adapter.read_response_from_db("multi", 2);
    assert_eq!(result, Some(("world\nline1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db("multi", 4);
    assert_eq!(result, Some(("line1\nline2\nline3".to_string(), 5)));

    let result = adapter.read_response_from_db("multi", 5);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
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
        skip_naration: false,
    };

    let result = adapter.read_response_from_db("empty", -1);
    assert_eq!(result, None);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_is_narration_true() {
    assert!(Adapter::is_narration("I will fetch the latest commits."));
    assert!(Adapter::is_narration("I'll fetch the latest commits."));
    assert!(Adapter::is_narration("I’ll fetch the latest commits."));
    assert!(Adapter::is_narration(
        "I will fetch the latest commits.\nI'll check the diff."
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
fn test_filter_narration_drops_all_narration() {
    let parts = vec![
        "I will fetch the latest commits.\nI will check the diff.".to_string(),
        "I will read the file.".to_string(),
        "The fix is confirmed! LGTM ✅".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result.as_deref(), Some("The fix is confirmed! LGTM ✅"));
}

#[test]
fn test_filter_narration_preserves_content_after_first_non_narration() {
    let parts = vec![
        "I will check things.".to_string(),
        "Here is my analysis.".to_string(),
        "I will also note this is fine.".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result.as_deref(), Some("Here is my analysis."));
}

#[test]
fn test_filter_narration_single_part_unchanged() {
    let parts = vec!["I will do something.".to_string()];
    let result = Adapter::filter_narration(&parts);
    assert_eq!(result, None);
}

#[test]
fn test_filter_narration_all_narration_drops_all() {
    let parts = vec![
        "I will fetch the file.".to_string(),
        "I'll check the output.".to_string(),
        "I will verify the fix.".to_string(),
    ];
    let result = filter_narration(&parts);
    assert_eq!(result, None);
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
    let config_options = result.get("configOptions").unwrap().as_array().unwrap();
    assert_eq!(config_options.len(), 1);
    assert_eq!(config_options[0]["id"].as_str(), Some("model"));
    assert_eq!(config_options[0]["category"].as_str(), Some("model"));
    assert_eq!(config_options[0]["type"].as_str(), Some("select"));
    assert!(config_options[0].get("currentValue").is_some());
    assert!(config_options[0].get("options").is_some());
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
fn test_session_set_config_option_sets_model() {
    let mut adapter = Adapter::new();
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let new_resp = adapter.handle_session_new(json!(1));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let set_resp = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "model", "value": "Model B"}),
    );

    assert!(set_resp.error.is_none(), "error: {:?}", set_resp.error);
    assert_eq!(
        adapter
            .sessions
            .get(&session_id)
            .unwrap()
            .model_id
            .as_deref(),
        Some("Model B")
    );
    let config_options = set_resp.result.as_ref().unwrap()["configOptions"]
        .as_array()
        .unwrap();
    assert_eq!(config_options[0]["currentValue"].as_str(), Some("Model B"));
}

#[test]
fn test_session_set_config_option_rejects_unknown_config() {
    let mut adapter = Adapter::new();
    let new_resp = adapter.handle_session_new(json!(1));
    let session_id = new_resp.result.as_ref().unwrap()["sessionId"]
        .as_str()
        .unwrap()
        .to_string();

    let resp = adapter.handle_session_set_config_option(
        json!(2),
        &json!({"sessionId": session_id, "configId": "not-model", "value": "Model B"}),
    );

    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap()["code"].as_i64(), Some(-32602));
}

#[test]
#[ignore]
fn test_session_set_model_persists() {
    let root = std::env::temp_dir().join(format!("agy-acp-model-persist-{}", Uuid::new_v4()));
    let _ = fs::create_dir_all(&root);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
    };

    adapter.persist_session("sess-m1", Some("conv-m1"), 0, None);

    adapter.restore_session_state("sess-m1");
    adapter.handle_session_set_model(
        json!(1),
        &json!({"sessionId": "sess-m1", "modelId": "Claude Opus 4.6 (Thinking)"}),
    );

    let adapter2 = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: root.join("conversations"),
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
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
    adapter.sessions.insert(
        "test-load".to_string(),
        crate::types::Session {
            conversation_id: None,
            last_step_idx: -1,
            model_id: Some("Gemini 3.1 Pro (High)".to_string()),
        },
    );
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
    assert_eq!(
        response["result"]["configOptions"][0]["currentValue"].as_str(),
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

    let response = adapter.handle_session_resume(json!(1), &json!({"sessionId": "test-resume"}));
    assert!(response.error.is_none(), "error: {:?}", response.error);
    let models = response.result.as_ref().unwrap()["models"]
        .as_object()
        .unwrap();
    assert_eq!(
        models["currentModelId"].as_str(),
        Some("GPT-OSS 120B (Medium)")
    );
    assert_eq!(
        response.result.as_ref().unwrap()["configOptions"][0]["currentValue"].as_str(),
        Some("GPT-OSS 120B (Medium)")
    );
}

#[test]
fn test_session_models_json_default() {
    let mut adapter = Adapter::new();
    let models = adapter.session_models_json(None);
    let current = models["currentModelId"].as_str().unwrap();
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

/// Build a synthetic generate_image step_payload mirroring real agy data:
///   top.1 varint = 91
///   top.5 (sub-msg) = tool call with name "generate_image" and title "Image generation"
///   top.104 (sub-msg) = image metadata
///     .1 prompt, .4 ImageName, .5 model
///     .6 (sub-msg) = media ref
///       .1 MIME, .5 file URI
fn make_image_step_payload(name: &str, mime: &str, file_uri: &str) -> Vec<u8> {
    let mut media = Vec::new();
    push_len_field(&mut media, 1, mime.as_bytes());
    push_len_field(&mut media, 5, file_uri.as_bytes());

    let mut image_meta = Vec::new();
    push_len_field(&mut image_meta, 1, b"a prompt");
    push_len_field(&mut image_meta, 4, name.as_bytes());
    push_len_field(&mut image_meta, 5, b"gemini-3.1-flash-image");
    push_len_field(&mut image_meta, 6, &media);

    let tool_payload = make_tool_payload(
        "img-call-1",
        "generate_image",
        &format!(r#"{{"ImageName":"{name}","toolSummary":"Image generation"}}"#),
        "Image generation",
        None,
    );

    let mut payload = tool_payload;
    push_len_field(&mut payload, 104, &image_meta);
    payload
}

#[test]
fn test_is_tool_step_type_includes_generate_image_step() {
    assert!(
        is_tool_step_type(91),
        "step_type 91 (generate_image) must be recognised as a tool step"
    );
}

#[test]
fn test_extract_image_artifact_from_step_payload_file_uri() {
    let payload = make_image_step_payload(
        "ui_mockup",
        "image/jpeg",
        "file:///tmp/generated.png",
    );
    let artifact = extract_image_artifact_from_step_payload(&payload)
        .expect("image artifact must be extracted");
    assert_eq!(artifact.absolute_path, "/tmp/generated.png");
    assert_eq!(artifact.mime.as_deref(), Some("image/jpeg"));
    assert_eq!(artifact.name.as_deref(), Some("ui_mockup"));
}

#[test]
fn test_extract_image_artifact_percent_decodes_file_uri_spaces() {
    // `file:///tmp/my%20image.png` must point to `/tmp/my image.png` on disk,
    // not the literal `/tmp/my%20image.png` which the OS would never find.
    let payload = make_image_step_payload(
        "spaced",
        "image/png",
        "file:///tmp/my%20image.png",
    );
    let artifact = extract_image_artifact_from_step_payload(&payload)
        .expect("percent-encoded file URI must decode");
    assert_eq!(artifact.absolute_path, "/tmp/my image.png");
}

#[test]
fn test_extract_image_artifact_percent_decodes_utf8_file_uri() {
    // `あ` is U+3042, encoded as %E3%81%82 in UTF-8 percent-encoding.
    let payload = make_image_step_payload(
        "japanese",
        "image/png",
        "file:///tmp/%E3%81%82.png",
    );
    let artifact = extract_image_artifact_from_step_payload(&payload)
        .expect("UTF-8 percent-encoded file URI must decode");
    assert_eq!(artifact.absolute_path, "/tmp/あ.png");
}

#[test]
fn test_extract_image_artifact_leaves_bare_absolute_path_unchanged() {
    // A bare absolute path that happens to contain `%` is NOT URL-encoded —
    // it must round-trip verbatim.
    let payload =
        make_image_step_payload("raw", "image/png", "/tmp/100%25 done.png");
    let artifact = extract_image_artifact_from_step_payload(&payload)
        .expect("bare absolute path must be accepted as-is");
    assert_eq!(artifact.absolute_path, "/tmp/100%25 done.png");
}

#[test]
fn test_extract_image_artifact_accepts_bare_absolute_path() {
    let payload = make_image_step_payload("banana", "image/png", "/var/folders/x/banana.jpg");
    let artifact = extract_image_artifact_from_step_payload(&payload)
        .expect("absolute path without file:// scheme must work");
    assert_eq!(artifact.absolute_path, "/var/folders/x/banana.jpg");
}

#[test]
fn test_extract_image_artifact_rejects_non_image_extension() {
    let payload = make_image_step_payload("note", "text/plain", "file:///tmp/note.txt");
    assert!(
        extract_image_artifact_from_step_payload(&payload).is_none(),
        "non-image extensions must be rejected"
    );
}

#[test]
fn test_extract_image_artifact_returns_none_without_field_104() {
    let payload = make_tool_payload(
        "non-image",
        "view_file",
        r#"{"AbsolutePath":"/tmp/x"}"#,
        "View file",
        None,
    );
    assert!(extract_image_artifact_from_step_payload(&payload).is_none());
}

#[test]
fn test_image_markdown_message_formats_link() {
    assert_eq!(
        image_markdown_message("/tmp/generated.png"),
        "![Generated image](/tmp/generated.png)"
    );
}

#[test]
fn test_image_markdown_message_escapes_closing_paren() {
    // A path containing `)` would prematurely close the markdown URL part.
    assert_eq!(
        image_markdown_message("/tmp/weird (1).png"),
        r"![Generated image](/tmp/weird (1\).png)"
    );
}

#[test]
fn test_image_markdown_message_escapes_backslash() {
    // A literal backslash should be doubled so the markdown URL stays valid.
    assert_eq!(
        image_markdown_message(r"/tmp/odd\path.png"),
        r"![Generated image](/tmp/odd\\path.png)"
    );
}

/// A 1×1 transparent PNG, base64-encoded — small enough to inline for tests.
const PNG_1X1_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII=";

#[test]
#[ignore]
fn test_materialize_data_uri_image_writes_png_to_dir() {
    let dir = std::env::temp_dir().join(format!("agy-acp-mat-{}", Uuid::new_v4()));
    let path = materialize_data_uri_image(PNG_1X1_DATA_URI, &dir)
        .expect("data uri must materialize to a path");
    assert!(path.ends_with(".png"), "expected .png extension, got {path}");
    let bytes = std::fs::read(&path).unwrap();
    assert!(
        bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "materialized file must contain a real PNG header"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_materialize_data_uri_image_rejects_non_image_mime() {
    let dir = std::env::temp_dir().join(format!("agy-acp-mat-rej-{}", Uuid::new_v4()));
    // text/plain is not an image — should be refused even if base64 is valid.
    let uri = "data:text/plain;base64,aGVsbG8=";
    assert!(materialize_data_uri_image(uri, &dir).is_none());
}

#[test]
fn test_materialize_data_uri_image_rejects_non_base64_payload() {
    let dir = std::env::temp_dir().join(format!("agy-acp-mat-raw-{}", Uuid::new_v4()));
    // Missing `;base64` marker — raw data URIs are not supported.
    let uri = "data:image/png,not-base64";
    assert!(materialize_data_uri_image(uri, &dir).is_none());
}

#[test]
#[ignore]
fn test_extract_image_artifact_materializes_data_uri() {
    // Scope materialization to $TMPDIR so the test never touches ~/.openab/agy-acp.
    let dir = std::env::temp_dir().join(format!("agy-acp-ext-{}", Uuid::new_v4()));
    let payload = make_image_step_payload("inline_img", "image/png", PNG_1X1_DATA_URI);
    let artifact = extract_image_artifact_from_step_payload_in_dir(&payload, &dir)
        .expect("data URI payload must yield an image artifact");
    assert!(
        artifact.absolute_path.starts_with(dir.to_str().unwrap()),
        "artifact must live under the injected tempdir, got {}",
        artifact.absolute_path
    );
    assert!(
        artifact.absolute_path.ends_with(".png"),
        "got {}",
        artifact.absolute_path
    );
    let bytes = std::fs::read(&artifact.absolute_path).unwrap();
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_extract_tool_update_for_generate_image_step() {
    let payload = make_image_step_payload(
        "ui_mockup",
        "image/jpeg",
        "file:///tmp/generated.png",
    );
    let update = extract_tool_update_from_step_payload(8, 91, &payload)
        .expect("step_type 91 must yield a tool_call update");
    assert_eq!(update["sessionUpdate"], "tool_call");
    assert_eq!(update["toolCallId"], "img-call-1");
    assert_eq!(update["title"], "Image generation");
}

#[test]
#[ignore]
fn test_streaming_emits_generate_image_markdown_chunk_only_once() {
    let root = std::env::temp_dir().join(format!("agy-acp-stream-img-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let db_path = conv_dir.join("conv-stream.db");
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
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 91, ?2)",
        rusqlite::params![
            3i64,
            make_image_step_payload(
                "banner",
                "image/png",
                "file:///tmp/agy-acp-test/banner.png",
            )
        ],
    )
    .unwrap();
    drop(conn);

    let state = Arc::new(Mutex::new(StreamingState {
        conversation_id: Some("conv-stream".to_string()),
        base_step_idx: -1,
        last_step_idx: -1,
        had_updates: false,
        agent_text_lengths: HashMap::new(),
        emitted_tool_steps: std::collections::HashSet::new(),
        emitted_image_steps: std::collections::HashSet::new(),
        last_title: None,
        skip_naration: false,
    }));

    let first = poll_streaming_delta(&conv_dir, None, "sess-x", &state);
    let second = poll_streaming_delta(&conv_dir, None, "sess-x", &state);

    fn collect_image_chunks(lines: &[String]) -> Vec<String> {
        lines
            .iter()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|n| n["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
            .filter_map(|n| {
                n["params"]["update"]["content"]["text"]
                    .as_str()
                    .map(String::from)
            })
            .filter(|text| text.contains("![Generated image]"))
            .collect()
    }

    let first_chunks = collect_image_chunks(&first);
    let second_chunks = collect_image_chunks(&second);

    assert_eq!(
        first_chunks.len(),
        1,
        "first poll should emit exactly one image markdown chunk"
    );
    assert!(
        first_chunks[0].contains("![Generated image](/tmp/agy-acp-test/banner.png)"),
        "image chunk should contain the markdown link"
    );
    assert_eq!(
        second_chunks.len(),
        0,
        "second poll should NOT re-emit the same image"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
#[ignore]
fn test_session_load_replays_generate_image_as_markdown_chunk() {
    let root = std::env::temp_dir().join(format!("agy-acp-image-{}", Uuid::new_v4()));
    let conv_dir = root.join("conversations");
    fs::create_dir_all(&conv_dir).unwrap();

    let db_path = conv_dir.join("conv-image.db");
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
        rusqlite::params![1i64, make_user_payload("draw a banana")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 15, ?2)",
        rusqlite::params![2i64, make_assistant_payload("Sure, generating now.")],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?1, 91, ?2)",
        rusqlite::params![
            3i64,
            make_image_step_payload(
                "happy_banana",
                "image/jpeg",
                "file:///tmp/agy-acp-test/happy_banana.png",
            )
        ],
    )
    .unwrap();
    drop(conn);

    let mut adapter = Adapter {
        sessions: HashMap::new(),
        working_dir: root.to_string_lossy().to_string(),
        conversations_dir: conv_dir,
        state_file: root.join("sessions.json"),
        available_models: vec![],
        skip_naration: false,
    };
    adapter.persist_session("sess-img", Some("conv-image"), -1, None);

    let output = adapter.handle_session_load(json!(1), &json!({"sessionId": "sess-img"}));
    let updates: Vec<Value> = output[..output.len() - 1]
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();

    let session_updates: Vec<&str> = updates
        .iter()
        .map(|n| n["params"]["update"]["sessionUpdate"].as_str().unwrap())
        .collect();

    // Expected sequence: user, assistant text (flushed), tool_call, agent_message_chunk (image).
    assert_eq!(
        session_updates,
        vec![
            "user_message_chunk",
            "agent_message_chunk",
            "tool_call",
            "agent_message_chunk",
        ],
        "pending assistant text must be flushed before the image is emitted"
    );

    let image_chunk = updates
        .last()
        .expect("expected an image markdown chunk at the end");
    assert_eq!(
        image_chunk["params"]["update"]["content"]["text"]
            .as_str()
            .unwrap()
            .trim(),
        "![Generated image](/tmp/agy-acp-test/happy_banana.png)"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn test_session_config_options_json_with_model() {
    let mut adapter = Adapter::new();
    adapter.available_models = vec!["Model A".to_string(), "Model B".to_string()];
    let config_options = adapter.session_config_options_json(Some("Model B"));
    assert_eq!(config_options[0]["id"].as_str(), Some("model"));
    assert_eq!(config_options[0]["category"].as_str(), Some("model"));
    assert_eq!(config_options[0]["type"].as_str(), Some("select"));
    assert_eq!(config_options[0]["currentValue"].as_str(), Some("Model B"));
    let options = config_options[0]["options"].as_array().unwrap();
    assert_eq!(options.len(), 2);
    assert_eq!(options[0]["value"].as_str(), Some("Model A"));
    assert_eq!(options[1]["value"].as_str(), Some("Model B"));
}
