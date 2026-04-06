from unittest.mock import MagicMock, patch

import httpx
import json

import llm
import pytest
import respx
from llm.plugins import load_plugins, pm
from mcp.types import CallToolResult, TextContent

from llm_devin import DeepWikiModel, DevinModel

ORG_ID = "org-test123"
BASE_URL = "https://api.devin.ai/v3"


def test_plugin_is_installed():
    load_plugins()

    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_devin" in names


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_execute_flow(monkeypatch, respx_mock):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    respx_mock.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        headers__contains={"Authorization": "Bearer test-api-key"},
        json__eq={"prompt": "Hello. How are you?"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
        headers__contains={"Authorization": "Bearer test-api-key"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "waiting_for_user",
            },
        )
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        headers__contains={"Authorization": "Bearer test-api-key"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "items": [
                    {
                        "event_id": "evt-1",
                        "source": "user",
                        "message": "Hello. How are you?",
                        "created_at": 1000,
                    },
                    {
                        "event_id": "evt-2",
                        "source": "devin",
                        "message": "Hello! I'm doing well, thank you for asking. How can I assist you today?",
                        "created_at": 1001,
                    },
                ],
                "end_cursor": "cursor-1",
                "has_next_page": False,
            },
        )
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello. How are you?"

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=MagicMock(),
            key="test-api-key",
        )
    )

    assert len(actual) == 1
    assert (
        actual[0]
        == "Hello! I'm doing well, thank you for asking. How can I assist you today?"
    )


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_execute_flow_exit_status(monkeypatch, respx_mock):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    respx_mock.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        headers__contains={"Authorization": "Bearer test-api-key"},
        json__eq={"prompt": "Fix the bug"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
        headers__contains={"Authorization": "Bearer test-api-key"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "status": "exit",
                "status_detail": None,
            },
        )
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        headers__contains={"Authorization": "Bearer test-api-key"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "items": [
                    {
                        "event_id": "evt-1",
                        "source": "devin",
                        "message": "Done!",
                        "created_at": 1000,
                    },
                ],
                "end_cursor": None,
                "has_next_page": False,
            },
        )
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Fix the bug"

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=MagicMock(),
            key="test-api-key",
        )
    )

    assert actual == ["Done!"]


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_execute_flow_multi_page_messages(monkeypatch, respx_mock):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    respx_mock.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        headers__contains={"Authorization": "Bearer test-api-key"},
        json__eq={"prompt": "Do something"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
        headers__contains={"Authorization": "Bearer test-api-key"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "status": "exit",
                "status_detail": None,
            },
        )
    )
    page1 = httpx.Response(
        status_code=200,
        json={
            "items": [
                {
                    "event_id": "evt-1",
                    "source": "devin",
                    "message": "Page 1 message",
                    "created_at": 1000,
                },
            ],
            "end_cursor": "cursor-after-page1",
            "has_next_page": True,
        },
    )
    page2 = httpx.Response(
        status_code=200,
        json={
            "items": [
                {
                    "event_id": "evt-2",
                    "source": "devin",
                    "message": "Page 2 message",
                    "created_at": 1001,
                },
            ],
            "end_cursor": "cursor-after-page2",
            "has_next_page": False,
        },
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        headers__contains={"Authorization": "Bearer test-api-key"},
    ).mock(side_effect=[page1, page2])

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Do something"

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=MagicMock(),
            key="test-api-key",
        )
    )

    assert actual == ["Page 1 message", "\nPage 2 message"]


def test_execute_requires_org_id(monkeypatch):
    monkeypatch.delenv("LLM_DEVIN_ORG_ID", raising=False)

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello"

    with pytest.raises(llm.ModelError, match="LLM_DEVIN_ORG_ID"):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=MagicMock(),
                key="test-api-key",
            )
        )


@patch("llm_devin.DeepWikiClient.run")
def test_deepwiki_execute(client_run):
    client_run.return_value = CallToolResult(
        isError=False,
        content=[
            TextContent(
                type="text", text="DeepWiki markdown for repository ftnext/llm-devin"
            )
        ],
    )

    sut = DeepWikiModel()
    prompt = MagicMock()
    prompt.prompt = "Summarize this repository."
    prompt.options.repository = "ftnext/llm-devin"

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=MagicMock(),
        )
    )

    assert len(actual) == 1
    assert actual[0] == "DeepWiki markdown for repository ftnext/llm-devin"


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_debug_logging_creates_jsonl_file(monkeypatch, respx_mock, tmp_path):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)

    create_session_data = {
        "session_id": "devin-test-session",
        "url": "https://app.devin.ai/sessions/devin-test-session",
        "status": "running",
    }
    session_detail_data = {
        "session_id": "devin-test-session",
        "status": "running",
        "status_detail": "finished",
    }
    messages_data = {
        "items": [
            {
                "event_id": "evt-1",
                "source": "devin",
                "message": "Done!",
                "created_at": 1000,
            },
        ],
        "end_cursor": None,
        "has_next_page": False,
    }
    respx_mock.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
    ).mock(return_value=httpx.Response(200, json=create_session_data))
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(return_value=httpx.Response(200, json=session_detail_data))
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(return_value=httpx.Response(200, json=messages_data))

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options.debug = True

    list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=MagicMock(),
            key="test-api-key",
        )
    )

    log_dir = tmp_path / "devin"
    assert log_dir.exists()
    jsonl_files = list(log_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    lines = jsonl_files[0].read_text().strip().splitlines()
    records = [json.loads(line) for line in lines]

    messages = [r["message"] for r in records]
    assert "create_session response" in messages
    assert "get_session response" in messages
    assert "messages response" in messages

    create_record = next(
        r for r in records if r["message"] == "create_session response"
    )
    assert create_record["data"] == create_session_data

    session_record = next(
        r for r in records if r["message"] == "get_session response"
    )
    assert session_record["data"] == session_detail_data

    messages_record = next(
        r for r in records if r["message"] == "messages response"
    )
    assert messages_record["data"] == messages_data

    for record in records:
        assert "timestamp" in record


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_no_debug_logging_when_debug_option_is_false(
    monkeypatch, respx_mock, tmp_path
):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)

    respx_mock.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "exit",
                "status_detail": None,
            },
        )
    )
    respx_mock.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "event_id": "evt-1",
                        "source": "devin",
                        "message": "Done!",
                        "created_at": 1000,
                    },
                ],
                "end_cursor": None,
                "has_next_page": False,
            },
        )
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options.debug = False

    list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=MagicMock(),
            key="test-api-key",
        )
    )

    log_dir = tmp_path / "devin"
    assert not log_dir.exists()
