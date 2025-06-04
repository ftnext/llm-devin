from unittest.mock import MagicMock, patch

import httpx
import respx
from llm.plugins import load_plugins, pm
from mcp.types import CallToolResult, TextContent

from llm_devin import DeepWikiModel, DevinModel


def test_plugin_is_installed():
    load_plugins()

    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_devin" in names


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_execute_flow(respx_mock):
    respx_mock.post(
        "https://api.devin.ai/v1/sessions",
        headers__contains={"Authorization": "Bearer test-api-key"},
        json__eq={"prompt": "Hello. How are you?", "idempotent": True},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
            },
        )
    )
    respx_mock.get(
        "https://api.devin.ai/v1/session/devin-test-session",
        headers__contains={"Authorization": "Bearer test-api-key"},
    ).mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "status_enum": "blocked",
                "messages": [
                    {
                        "type": "initial_user_message",
                        "message": "Hello. How are you?",
                    },
                    {
                        "type": "devin_message",
                        "message": "Hello! I'm doing well, thank you for asking. How can I assist you today?",
                    },
                ],
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
