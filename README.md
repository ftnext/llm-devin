# llm-devin

[![PyPI](https://img.shields.io/pypi/v/llm-devin.svg)](https://pypi.org/project/llm-devin/)
[![Changelog](https://img.shields.io/github/v/release/ftnext/llm-devin?include_prereleases&label=changelog)](https://github.com/ftnext/llm-devin/releases)
[![Tests](https://github.com/ftnext/llm-devin/actions/workflows/test.yml/badge.svg)](https://github.com/ftnext/llm-devin/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ftnext/llm-devin/blob/main/LICENSE)



## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-devin
```
## Usage

### Devin API

**prerequisite**: Devin API key and Organization ID (Devin Team Plan)  
https://docs.devin.ai/api-reference/overview

Set up a service user and get your organization ID from **Settings > Service users**.

```bash
export LLM_DEVIN_KEY=your_api_key_here
export LLM_DEVIN_ORG_ID=your_org_id_here

llm -m devin "Hello, Devin"
```

Options for creating a new session (all optional; see the [API reference](https://docs.devin.ai/api-reference/v3/sessions/post-organizations-sessions)):

```bash
llm -m devin \
  -o title "Release notes" \
  -o tags "release-notes,owner-repo" \
  -o repos "owner/repo" \
  -o max_acu_limit 5 \
  -o playbook_id playbook-xxxx \
  -o devin_mode fast \
  "Explain the changes in the latest release of https://github.com/owner/repo"
```

`tags` and `repos` are comma-separated. These options only apply when a new session is created; they are ignored when continuing a conversation.

Continue that Devin conversation with `llm -c` immediately after the previous command, or specify the model explicitly:

```bash
llm -m devin -c "Follow-up message"
```

Send a message to an existing Devin session by ID or URL, even when there is no local `llm` conversation history for it (e.g. a session started from the Devin web app or on another machine):

```bash
llm -m devin -o session devin-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx "Follow-up message"
llm -m devin -o session https://app.devin.ai/sessions/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx "Follow-up message"
```

Before sending, `Continuing Devin session: <URL>` is printed. If the value is not a valid session ID/URL, or the session does not exist or is not accessible with your API key, the command fails without creating a new session. The session is recorded in the `llm` log, so you can keep going with `llm -c` afterwards.

`session` cannot be combined with `-c`/`--cid` when that conversation already has history for a different (or unknown) Devin session, because the destination would be ambiguous; such a command is rejected. Use `-c`/`--cid` alone to continue the logged conversation, or `-o session` alone to target a specific session. If the logged conversation already belongs to the same session, the combination is allowed (this is what `llm chat -m devin -o session ...` does on every turn after the first).

Start an interactive chat session:

```bash
llm chat -m devin
```

### Reading an existing Devin session

`llm devin status` and `llm devin messages` read an existing session without sending anything to Devin. They only call read-only Devin API endpoints (get session, list session messages): no session is created, no message is sent, and no suspended session is resumed. They need no local `llm` conversation history, so they work after the CLI exited or the connection dropped, and for sessions started from the Devin web app or another machine. Each command fetches once and exits; it does not poll until the session finishes.

Both commands accept a session ID or an `app.devin.ai` session URL (the same values as `-o session`), and read `LLM_DEVIN_KEY` (or `llm keys set devin`, or `--key`) and `LLM_DEVIN_ORG_ID`.

```bash
llm devin status devin-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
llm devin status https://app.devin.ai/sessions/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
llm devin status devin-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --json
```

`status` shows what the API returns for the session: status and status detail, title, tags, creation/update timestamps (raw values from the API), consumed ACUs, structured output, pull request URLs with their state, plus the latest message from Devin. Fields the API does not provide are omitted.

```bash
llm devin messages devin-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
llm devin messages devin-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --source devin --limit 3
llm devin messages devin-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --json
```

`messages` lists the session messages oldest first, following the API pagination to the end so that `--limit N` returns the N most recent ones. `--source devin|user` filters by sender.

With `--json`, only JSON is written to standard output, so it can be piped into `jq` or another script. Invalid session IDs/URLs, a missing key or organization ID, sessions that are not accessible, and API or network failures are reported on standard error and exit with a non-zero status.

### DeepWiki

```bash
llm -m deepwiki -o repository ftnext/llm-devin "Summarize this repository"
```

## Development

To set up this plugin locally, first checkout the code:
```bash
cd llm-devin
```
Then create a new virtual environment and install the dependencies and test dependencies:
```bash
uv sync --extra test
```
To run the tests:
```bash
uv run pytest
```
