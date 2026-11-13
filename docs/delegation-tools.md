# Delegating work to Devin from other models (`Devin` toolbox)

llm-devin registers a `Devin` [toolbox](https://llm.datasette.io/en/stable/python-api.html#toolbox-classes) through the [`register_tools`](https://llm.datasette.io/en/stable/plugins/plugin-hooks.html#register-tools-register) hook.
A tool-capable model (for example an OpenAI or Gemini model in `llm`) can then hand a task off to a cloud Devin session, and later read its status and messages.

Tools require LLM 0.26 or later. With older versions the `devin` and `deepwiki` models keep working; only the toolbox is unavailable.

## Tools

| Tool | What it does | Devin API |
|------|--------------|-----------|
| `Devin_start_session` | Starts **one new** cloud Devin session and returns its ID and URL immediately (no waiting, no polling). | `POST /v3/organizations/{org_id}/sessions` |
| `Devin_session_status` | Reads the session status once: status, status detail, ACUs consumed, pull requests, structured output. Read-only. | `GET /v3/organizations/{org_id}/sessions/{devin_id}` |
| `Devin_session_messages` | Reads the most recent messages (default: 5 from Devin, at most 20, each truncated to 4000 characters). Read-only. | `GET /v3/organizations/{org_id}/sessions/{devin_id}/messages` |

The handoff is asynchronous: Devin works for minutes or hours, so the calling model is expected to report the session URL to you and check back later, not to wait.
While the session is still working, `Devin_session_status` returns a `note` telling the model not to poll in a loop.
Malformed API responses (for example a status without `status`, or a message without `source`/`message`) are returned as errors rather than partial results.
As a stricter completeness check than the published schema (like `llm devin messages`), every message page must also include a boolean `has_next_page`, and a repeated page cursor is an error.
The read-only tools never create, message, or resume a session, and an invalid or inaccessible session is an error (there is no fallback to creating a new session).
To send follow-up instructions, use `llm -m devin -o session <ID or URL> "..."` yourself.

## Setup

The same credentials as the `devin` model are used: `LLM_DEVIN_KEY` (or `llm keys set devin`) and `LLM_DEVIN_ORG_ID`.
The key is read when a tool runs; it is never a tool argument or a toolbox option, so it is not sent to the calling model or stored in the `llm` logs.

## Usage

```bash
export LLM_DEVIN_KEY=your_api_key_here
export LLM_DEVIN_ORG_ID=your_org_id_here

llm -m gpt-4.1-mini -T Devin --td --ta \
  "Ask Devin to add a regression test for issue #12 in owner/repo, then tell me the session URL"

# later, read-only
llm -m gpt-4.1-mini -T Devin --td \
  "What is the status of devin-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx? Summarize its latest messages"
```

- `-T Devin` (`--tool`) makes the toolbox available to the model. `llm tools list` shows the tools and their descriptions.
- `--td` (`--tools-debug`) prints each tool call and its JSON result, so you can see the session ID, URL and applied `max_acu_limit`.
- `--ta` (`--tools-approve`) asks for confirmation before every tool call. It is recommended, because starting a session spends ACUs.
- `llm chat -m gpt-4.1-mini -T Devin` works as well. Errors (invalid input, missing credentials, API failures) are returned to the model as `Error: ...` tool results.

### Configuration

Options are passed in the toolbox specification:

```bash
llm -m gpt-4.1-mini -T 'Devin(max_acu_limit=3, devin_mode="fusion")' --td --ta "..."
```

| Option | Default | Meaning |
|--------|---------|---------|
| `max_acu_limit` | `5` | ACU cap sent as `max_acu_limit` on every session this toolbox starts (1-50). The model may request a lower cap, never a higher one. |
| `devin_mode` | unset | Fixes the agent mode. When unset, the model may choose one of the modes below, or omit it to use the organization default. |
| `max_sessions` | `1` | How many sessions one toolbox instance (one `llm` prompt or one `llm chat`) may start (1-10). |

A start attempt rejected with an HTTP 4xx error (for example an invalid key or invalid repository) does not count toward `max_sessions`.
Any other failure counts. A timeout or interrupted connection after the request was sent, an HTTP 5xx or an unexpected response is reported as "Could not confirm whether a Devin session was started", because the session may already exist and consume ACUs; check the Devin web app before retrying.
Invalid options raise `ValueError` when the toolbox is created.

## Agent modes (Fusion, SWE-2)

`devin_mode` accepts this subset of the [documented values](https://docs.devin.ai/api-reference/v3/sessions/post-organizations-sessions): `normal`, `lite`, `fusion`, `swe-2-medium`, `swe-2-high`, `swe-2-max`.
`fast`, `fast-ultra` and `ultra` are intentionally not available to the calling model. Use `llm -m devin -o devin_mode fast` or `-o devin_mode ultra` directly if you need them.
According to the API reference, modes other than `normal`, `fast`, `lite`, `ultra` and `fusion` select the web app agent mode of the same name, and preview modes are subject to the same feature flags and enterprise restrictions as the web app.

These are **agent modes**, not raw model inference: every call starts a full Devin session (VM, repository access, planning, PRs) billed in ACUs.
The Devin v3 API reference does not document a public endpoint for raw SWE-2 or SWE-2 Flash completions, so this toolbox cannot be used as a cheap text model for another LLM. It is meant for delegating whole tasks, for example a "sidekick" pattern where a general model plans and hands implementation to Devin running Fusion or SWE-2.

## Cost bound vs. actual cost

- **Cost bound**: `max_acu_limit` is the upper limit you allow per session (default 5 ACUs), and `max_sessions` limits how many sessions one toolbox instance starts. Together they bound the ACUs a single prompt can commit.
- **Actual cost**: what a session really used is reported as `acus_consumed` by `Devin_session_status`, and may be well below the cap. Billing follows your Devin plan.
- **The calling model's cost** is separate: each tool call and result adds tokens to the conversation with the model you run in `llm` (e.g. OpenAI), billed by that provider. `--chain-limit` (default 5) bounds how many tool-call rounds one prompt can make.

## Python API

```python
import llm
from llm_devin import Devin

model = llm.get_model("gpt-4.1-mini")
devin = Devin(max_acu_limit=3, devin_mode="swe-2-high")
response = model.chain(
    "Ask Devin to fix the failing CI in owner/repo and give me the URL",
    tools=[devin],
)
print(response.text())
```

## Security notes

The calling model decides when to call the tools, so prompt injection in the content it reads could make it start a session. Use `--ta`, a low `max_acu_limit`, and only give the toolbox to models and inputs you trust.
The read tools can read any session the API key can view, as `llm devin status` / `llm devin messages` do.
DeepWiki is not part of this toolbox; use `llm -m deepwiki`.
