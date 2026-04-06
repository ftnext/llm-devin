from unittest.mock import MagicMock, patch

import httpx
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
