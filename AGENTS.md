# AGENTS.md

## Project Overview

`llm-devin` is an [LLM](https://llm.datasette.io/) plugin that provides two models:

- **`devin`**: Interacts with the [Devin API (v3)](https://docs.devin.ai/api-reference/overview) to create sessions and stream messages.
- **`deepwiki`**: Queries repositories via the [DeepWiki MCP server](https://mcp.deepwiki.com).

Single-file plugin: all logic lives in `llm_devin.py`.

## Setup

```bash
uv sync --extra test
```

## Running Tests

```bash
uv run pytest
```

Tests use [respx](https://lundberg.github.io/respx/) to mock HTTP calls and `monkeypatch` to set environment variables.

## Code Conventions

- Python 3.10+
- Minimal comments; if needed, write them in English.
- Keep `llm_devin.py` as a single module (no subpackages).

## Devin API

This plugin uses the **Devin API v3** (Organization scope).

- Base URL: `https://api.devin.ai/v3/organizations/{org_id}/...`
- Authentication: `Authorization: Bearer <key>` (service user token or legacy API key)
- Environment variables: `LLM_DEVIN_KEY` (API key), `LLM_DEVIN_ORG_ID` (organization ID)
- API reference: https://docs.devin.ai/api-reference/overview
- API release notes: https://docs.devin.ai/api-reference/release-notes

### Key Endpoints Used

| Operation       | Endpoint                                                        |
|-----------------|-----------------------------------------------------------------|
| Create session  | `POST /v3/organizations/{org_id}/sessions`                      |
| Get session     | `GET /v3/organizations/{org_id}/sessions/{devin_id}`            |
| List messages   | `GET /v3/organizations/{org_id}/sessions/{devin_id}/messages`   |

### Session Status Values

- `status`: `new`, `creating`, `claimed`, `running`, `exit`, `error`, `suspended`, `resuming`
- `status_detail` (when running): `working`, `waiting_for_user`, `waiting_for_approval`, `finished`

## CI

- GitHub Actions: `.github/workflows/test.yml` runs `pytest` on Python 3.10–3.14.
- Publishing: `.github/workflows/publish.yml` builds and publishes to PyPI on GitHub release.
