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

`session` cannot be combined with `-c`/`--cid` when that conversation already has history, because the destination would be ambiguous; such a command is rejected. Use `-c`/`--cid` alone to continue the logged conversation, or `-o session` alone to target a specific session.

Start an interactive chat session:

```bash
llm chat -m devin
```

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
