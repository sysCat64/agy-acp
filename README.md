# agy-acp

`agy-acp` is an [Agent Client Protocol (ACP)](https://agentclientprotocol.com)
stdio adapter for [Google Antigravity CLI](https://github.com/google-antigravity/antigravity-cli)
(`agy`).

It lets an ACP-compatible host, such as Paseo or Zed, talk to `agy` over
JSON-RPC. The adapter starts `agy` as a subprocess, reads Antigravity's local
conversation database, and turns the result into ACP `session/update`
notifications.

## How It Works

```text
ACP host  <--stdin/stdout JSON-RPC-->  agy-acp  <--subprocess-->  agy  <--API-->  Gemini
```

`agy-acp` is intentionally a thin compatibility layer. It does not change
`agy` itself into a native ACP server; it bridges the current CLI behavior and
Antigravity's local conversation data into the ACP shape expected by the host.

## Current Features

- ACP `initialize`, `session/new`, `session/load`, `session/resume`, and
  `session/prompt` support.
- Incremental `session/update` notifications while `agy` is running.
- Session persistence in `~/.openab/agy-acp/sessions.json`.
- Conversation replay from `~/.gemini/antigravity-cli/conversations/*.db`.
- Model listing via `agy models` and ACP model/config option responses.
- Generated image artifacts are emitted as Markdown image links when local file
  paths or inline data URIs can be extracted from Antigravity conversation data.

## Prerequisites

- Rust with Cargo.
- `agy` installed and available in `PATH`.
- Antigravity/Gemini authentication configured for `agy`, for example with
  `GEMINI_API_KEY` or `~/.gemini/antigravity-cli/settings.json`.

## Build

```bash
cargo build --release
```

The binary is created at `target/release/agy-acp`. You can reference that path
directly from your ACP host, or copy it somewhere in your `PATH`.

```bash
cp target/release/agy-acp /usr/local/bin/
```

## Host Configuration

Configure your ACP host to spawn the `agy-acp` binary over stdio. The exact
settings format depends on the host.

### Paseo

Example custom provider configuration:

```json
{
  "antigravity-acp": {
    "extends": "acp",
    "label": "Antigravity",
    "command": ["/absolute/path/to/agy-acp"],
    "params": {
      "supportsMcpServers": false
    }
  }
}
```

Use the absolute path to the built binary, such as
`/Users/you/path/to/agy-acp/target/release/agy-acp`, or a path in your `PATH`.

### Zed

Example custom agent server configuration:

```json
{
  "agent_servers": {
    "agy": {
      "type": "custom",
      "command": "agy-acp",
      "args": [],
      "env": {}
    }
  }
}
```

## Options

### Extra `agy` Arguments

Set `AGY_EXTRA_ARGS` to pass additional space-separated arguments to every
`agy` invocation:

```json
{
  "AGY_EXTRA_ARGS": "--some-flag value"
}
```

### Skip Narration

Run the adapter with `--skip-naration` to drop leading narration-only assistant
chunks such as "I will ...".

```bash
agy-acp --skip-naration
```

The option name keeps the current historical spelling.

## Generated Images

When Antigravity records a generated image artifact, `agy-acp` tries to extract
the local image reference from the conversation payload.

- `file://...` image URIs are converted to absolute local paths.
- Bare absolute image paths are used as-is.
- `data:image/...;base64,...` payloads are materialized under
  `~/.openab/agy-acp/images`.

The adapter sends the image back as assistant text containing Markdown:

```markdown
![Generated image](/absolute/path/to/generated.png)
```

Whether the image renders inline depends on the ACP host's Markdown/media
support.

This path has been verified end-to-end with Paseo: an image-generation prompt
sent through a custom ACP provider can invoke Antigravity and render the
generated local image inline in Paseo.

## Data Locations

| Path | Purpose |
|---|---|
| `~/.openab/agy-acp/sessions.json` | Persisted ACP session to Antigravity conversation mapping |
| `~/.openab/agy-acp/images/` | Materialized inline image data |
| `~/.gemini/antigravity-cli/conversations/*.db` | Antigravity conversation SQLite databases read by the adapter |

## Development

```bash
cargo build
cargo test
cargo test -- --include-ignored
cargo test e2e -- --ignored --nocapture
```

The e2e tests require a release build, an `agy` binary, and valid
Antigravity/Gemini authentication.

## Notes

`agy-acp` depends on Antigravity CLI's current local conversation storage format.
If `agy` gains native ACP support or changes its internal SQLite/protobuf layout,
this adapter may need to be updated.

## License

MIT
