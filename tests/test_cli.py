import json

import httpx2
import pytest
from click.testing import CliRunner
from llm.cli import cli

API_KEY = "test-api-key"
ORG_ID = "org-test123"
BASE_URL = "https://api.devin.ai/v3"
SESSION_HEX = "0123456789abcdef0123456789abcdef"
SESSION_ID = f"devin-{SESSION_HEX}"
SESSION_URL = f"https://app.devin.ai/sessions/{SESSION_HEX}"

SESSION_ENDPOINT = f"{BASE_URL}/organizations/{ORG_ID}/sessions/{SESSION_ID}"
MESSAGES_ENDPOINT = f"{SESSION_ENDPOINT}/messages"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setenv("LLM_DEVIN_KEY", API_KEY)


def session_response(**overrides):
    payload = {
        "session_id": SESSION_ID,
        "url": SESSION_URL,
        "status": "running",
        "status_detail": "working",
        "title": "Investigate the flaky test",
        "tags": ["ci"],
        "org_id": ORG_ID,
        "created_at": 1000,
        "updated_at": 2000,
        "acus_consumed": 1.5,
        "pull_requests": [
            {"pr_url": "https://github.com/owner/repo/pull/1", "pr_state": "open"}
        ],
        "structured_output": {"result": "ok"},
    }
    payload.update(overrides)
    return httpx2.Response(200, json=payload)


def messages_response(items, end_cursor=None, has_next_page=False):
    return httpx2.Response(
        200,
        json={
            "items": items,
            "end_cursor": end_cursor,
            "has_next_page": has_next_page,
        },
    )


def message(event_id, source, text, created_at):
    return {
        "event_id": event_id,
        "source": source,
        "message": text,
        "created_at": created_at,
    }


def mock_status(mock_cli_api, session=None, messages=None):
    mock_cli_api.get(SESSION_ENDPOINT).mock(
        return_value=session if session is not None else session_response()
    )
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response(
            messages
            if messages is not None
            else [
                message("evt-1", "user", "Please look into it", 1000),
                message("evt-2", "devin", "Found the cause", 1001),
            ]
        )
    )


@pytest.mark.parametrize(
    "reference",
    [SESSION_ID, SESSION_HEX, SESSION_URL],
    ids=["id", "bare-hex", "url"],
)
def test_status(env, mock_cli_api, reference):
    mock_status(mock_cli_api)

    result = CliRunner().invoke(cli, ["devin", "status", reference])

    assert result.exit_code == 0, result.output
    assert f"Session: {SESSION_ID}" in result.output
    assert f"URL: {SESSION_URL}" in result.output
    assert "Status: running (working)" in result.output
    assert "Title: Investigate the flaky test" in result.output
    assert "ACUs consumed: 1.5" in result.output
    assert "https://github.com/owner/repo/pull/1 (open)" in result.output
    assert "Found the cause" in result.output
    assert all(call.method == "GET" for call in mock_cli_api.calls)


def test_status_json(env, mock_cli_api):
    mock_status(mock_cli_api)

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID, "--json"])

    assert result.exit_code == 0, result.output
    actual = json.loads(result.output)
    assert actual["session"]["session_id"] == SESSION_ID
    assert actual["session"]["structured_output"] == {"result": "ok"}
    assert actual["latest_devin_message"] == message(
        "evt-2", "devin", "Found the cause", 1001
    )


def test_status_finished_session_without_messages_or_pull_requests(
    env, mock_cli_api
):
    mock_status(
        mock_cli_api,
        session=session_response(
            status="exit",
            status_detail=None,
            title=None,
            tags=[],
            pull_requests=[],
            structured_output=None,
        ),
        messages=[],
    )

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code == 0, result.output
    assert "Status: exit\n" in result.output
    assert "Pull requests: none" in result.output
    assert "Latest Devin message: none" in result.output
    assert "Title:" not in result.output
    assert "Structured output:" not in result.output


def test_status_latest_devin_message_comes_from_the_last_page(
    env, mock_cli_api
):
    mock_cli_api.get(SESSION_ENDPOINT).mock(return_value=session_response())
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        side_effect=[
            messages_response(
                [message("evt-1", "devin", "First answer", 1000)],
                end_cursor="cursor-1",
                has_next_page=True,
            ),
            messages_response(
                [message("evt-2", "devin", "Latest answer", 2000)],
                end_cursor="cursor-2",
            ),
        ]
    )

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID, "--json"])

    assert result.exit_code == 0, result.output
    actual = json.loads(result.output)
    assert actual["latest_devin_message"]["message"] == "Latest answer"
    messages_calls = [
        call for call in mock_cli_api.calls if "/messages" in str(call.url)
    ]
    assert [call.url.params.get("after") for call in messages_calls] == [
        None,
        "cursor-1",
    ]


def test_messages(env, mock_cli_api):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response(
            [
                message("evt-1", "user", "Please look into it", 1000),
                message("evt-2", "devin", "Found the cause", 1001),
            ]
        )
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_URL])

    assert result.exit_code == 0, result.output
    assert result.output == (
        "[user] Please look into it\n[devin] Found the cause\n"
    )
    assert all(call.method == "GET" for call in mock_cli_api.calls)


def test_messages_json_with_limit_and_source(env, mock_cli_api):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response(
            [
                message("evt-1", "devin", "First answer", 1000),
                message("evt-2", "user", "Thanks", 1001),
                message("evt-3", "devin", "Latest answer", 1002),
            ]
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "devin",
            "messages",
            SESSION_ID,
            "--json",
            "--source",
            "devin",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "session_id": SESSION_ID,
        "messages": [message("evt-3", "devin", "Latest answer", 1002)],
    }


def test_messages_without_messages(env, mock_cli_api):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response([])
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID])

    assert result.exit_code == 0, result.output
    assert result.output == "No messages\n"


def test_messages_json_without_messages_is_parsable(env, mock_cli_api):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response([])
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID, "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "session_id": SESSION_ID,
        "messages": [],
    }


@pytest.mark.parametrize(
    "command", ["status", "messages"], ids=["status", "messages"]
)
@pytest.mark.parametrize(
    "reference",
    ["not-a-session", "https://example.com/sessions/" + SESSION_HEX, "devin-xyz"],
    ids=["plain-text", "other-host", "invalid-id"],
)
def test_invalid_session_reference_makes_no_request(
    env, mock_cli_api, command, reference
):
    result = CliRunner().invoke(cli, ["devin", command, reference])

    assert result.exit_code != 0
    assert "Invalid Devin session" in result.output
    assert mock_cli_api.calls == []


def test_missing_org_id(monkeypatch, tmp_path, mock_cli_api):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.delenv("LLM_DEVIN_ORG_ID", raising=False)
    monkeypatch.setenv("LLM_DEVIN_KEY", API_KEY)

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code != 0
    assert "LLM_DEVIN_ORG_ID" in result.output
    assert mock_cli_api.calls == []


def test_missing_key(monkeypatch, tmp_path, mock_cli_api):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.delenv("LLM_DEVIN_KEY", raising=False)

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code != 0
    assert "No Devin API key found" in result.output
    assert mock_cli_api.calls == []


def test_unauthorized(env, mock_cli_api):
    mock_cli_api.get(SESSION_ENDPOINT).mock(
        return_value=httpx2.Response(401, json={"detail": "API key has expired"})
    )

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code != 0
    assert "authentication failed" in result.output
    assert "API key has expired" in result.output


@pytest.mark.parametrize("status_code", [403, 404])
def test_session_not_accessible(env, mock_cli_api, status_code):
    mock_cli_api.get(SESSION_ENDPOINT).mock(
        return_value=httpx2.Response(status_code, json={"detail": "Not found"})
    )

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code != 0
    assert f"HTTP {status_code}" in result.output
    assert "may not be accessible with this API key" in result.output


def test_server_error(env, mock_cli_api):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=httpx2.Response(500, text="boom")
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID])

    assert result.exit_code != 0
    assert "HTTP 500" in result.output


def test_connection_failure(env, mock_cli_api):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=httpx2.ConnectError("connection refused")
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID])

    assert result.exit_code != 0
    assert "Could not reach the Devin API" in result.output
    assert "ConnectError" in result.output


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, text="not json"),
        httpx2.Response(200, json=[]),
        httpx2.Response(200, json={"end_cursor": None}),
        httpx2.Response(200, json={"items": [], "end_cursor": None}),
        httpx2.Response(
            200, json={"items": [], "end_cursor": None, "has_next_page": "no"}
        ),
    ],
    ids=[
        "not-json",
        "not-object",
        "without-items",
        "without-has-next-page",
        "non-boolean-has-next-page",
    ],
)
def test_malformed_messages_response(env, mock_cli_api, response):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(return_value=response)

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID])

    assert result.exit_code != 0
    assert "unexpected response" in result.output


def test_malformed_session_response(env, mock_cli_api):
    mock_cli_api.get(SESSION_ENDPOINT).mock(
        return_value=httpx2.Response(200, text="not json")
    )

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code != 0
    assert "unexpected response" in result.output


def test_status_shows_empty_structured_output(env, mock_cli_api):
    mock_status(mock_cli_api, session=session_response(structured_output={}))

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code == 0, result.output
    assert "Structured output:\n{}" in result.output


def test_pagination_without_end_cursor(env, mock_cli_api):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response(
            [message("evt-1", "devin", "Answer", 1000)], has_next_page=True
        )
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID])

    assert result.exit_code != 0
    assert "without an end_cursor" in result.output


@pytest.mark.parametrize(
    "missing",
    ["session_id", "url", "status", "created_at", "updated_at", "acus_consumed"],
)
def test_session_response_missing_field(env, mock_cli_api, missing):
    payload = json.loads(session_response().content)
    del payload[missing]
    mock_cli_api.get(SESSION_ENDPOINT).mock(
        return_value=httpx2.Response(200, json=payload)
    )

    result = CliRunner().invoke(cli, ["devin", "status", SESSION_ID])

    assert result.exit_code != 0
    assert "unexpected response" in result.output


@pytest.mark.parametrize(
    "items",
    [
        [{"event_id": "evt-1", "created_at": 1000}],
        [{"event_id": "evt-1", "source": "devin", "created_at": 1000}],
        ["not an object"],
    ],
    ids=["without-source", "without-message", "not-object"],
)
def test_messages_response_missing_field(env, mock_cli_api, items):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response(items)
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID])

    assert result.exit_code != 0
    assert "unexpected response" in result.output


@pytest.mark.parametrize("end_cursor", [None, "", 1], ids=["null", "empty", "int"])
def test_pagination_with_unusable_end_cursor(env, mock_cli_api, end_cursor):
    mock_cli_api.get(MESSAGES_ENDPOINT).mock(
        return_value=messages_response(
            [message("evt-1", "devin", "Answer", 1000)],
            end_cursor=end_cursor,
            has_next_page=True,
        )
    )

    result = CliRunner().invoke(cli, ["devin", "messages", SESSION_ID])

    assert result.exit_code != 0
    assert "without an end_cursor" in result.output
