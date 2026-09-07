from unittest.mock import MagicMock, patch

import httpx2
import json

import llm
import pytest
from llm.plugins import load_plugins, pm
from mcp.types import CallToolResult, TextContent

from llm_devin import DeepWikiModel, DevinModel

ORG_ID = "org-test123"
BASE_URL = "https://api.devin.ai/v3"
API_KEY = "test-api-key"


class MockRoute:
    def __init__(self, method, url, json__eq, params__contains):
        self.method = method
        self.url = url
        self.json__eq = json__eq
        self.params__contains = params__contains
        self.responses = []
        self.called = False

    def mock(self, return_value=None, side_effect=None):
        self.responses = list(side_effect) if side_effect else [return_value]

    def matches(self, request):
        if request.method != self.method:
            return False
        if request.url.copy_with(query=None) != httpx2.URL(self.url):
            return False
        if self.json__eq is not None and json.loads(request.content) != self.json__eq:
            return False
        if self.params__contains is not None and any(
            request.url.params.get(k) != v for k, v in self.params__contains.items()
        ):
            return False
        return True

    def respond(self, request):
        self.called = True
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        response.request = request
        return response


class MockDevinApi:
    def __init__(self):
        self.routes = []
        self.calls = []

    def _route(self, method, url, json__eq=None, params__contains=None):
        route = MockRoute(method, url, json__eq, params__contains)
        self.routes.append(route)
        return route

    def get(self, url, **kwargs):
        return self._route("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._route("POST", url, **kwargs)

    def handler(self, request):
        self.calls.append(request)
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        for route in self.routes:
            if route.matches(request):
                return route.respond(request)
        pytest.fail(f"Unmocked request: {request.method} {request.url}")

    def assert_all_called(self):
        for route in self.routes:
            assert route.called, f"Route not called: {route.method} {route.url}"


@pytest.fixture
def mock_api():
    api = MockDevinApi()
    transport = httpx2.MockTransport(api.handler)

    def create_http_client(headers):
        return httpx2.Client(headers=headers, transport=transport)

    with patch("llm_devin._devin.create_http_client", create_http_client):
        yield api
    api.assert_all_called()


def test_plugin_is_installed():
    load_plugins()

    names = [mod.__name__ for mod in pm.get_plugins()]
    assert "llm_devin" in names


def test_execute_flow(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        json__eq={"prompt": "Hello. How are you?"},
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "waiting_for_user",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx2.Response(
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
    prompt.options = DevinModel.Options()

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=llm.Conversation(model=sut),
            key=API_KEY,
        )
    )

    assert len(actual) == 1
    assert (
        actual[0]
        == "Hello! I'm doing well, thank you for asking. How can I assist you today?"
    )


def test_execute_flow_exit_status(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        json__eq={"prompt": "Fix the bug"},
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "status": "exit",
                "status_detail": None,
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx2.Response(
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
    prompt.options = DevinModel.Options()

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=llm.Conversation(model=sut),
            key=API_KEY,
        )
    )

    assert actual == ["Done!"]


def test_create_session_with_options(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        json__eq={
            "prompt": "Explain the latest release",
            "title": "Release notes",
            "tags": ["release-notes", "owner-repo"],
            "repos": ["owner/repo"],
            "max_acu_limit": 5,
            "playbook_id": "playbook-abc",
            "devin_mode": "fast",
        },
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={"session_id": "devin-test-session", "status": "exit"},
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={"items": [], "end_cursor": None, "has_next_page": False},
        )
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Explain the latest release"
    prompt.options = DevinModel.Options(
        title="Release notes",
        tags="release-notes, owner-repo",
        repos="owner/repo",
        max_acu_limit=5,
        playbook_id="playbook-abc",
        devin_mode="fast",
    )

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=llm.Conversation(model=sut),
            key=API_KEY,
        )
    )

    assert actual == []


def test_execute_flow_multi_page_messages(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        json__eq={"prompt": "Do something"},
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "status": "exit",
                "status_detail": None,
            },
        )
    )
    page1 = httpx2.Response(
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
    page2 = httpx2.Response(
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
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(side_effect=[page1, page2])

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Do something"
    prompt.options = DevinModel.Options()

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=llm.Conversation(model=sut),
            key=API_KEY,
        )
    )

    assert actual == ["Page 1 message", "\nPage 2 message"]


def test_execute_requires_org_id(monkeypatch):
    monkeypatch.delenv("LLM_DEVIN_ORG_ID", raising=False)

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = DevinModel.Options()

    with pytest.raises(llm.ModelError, match="LLM_DEVIN_ORG_ID"):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=llm.Conversation(model=sut),
                key=API_KEY,
            )
        )


@patch("llm_devin._deepwiki.DeepWikiClient.run")
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


def test_debug_logging_creates_jsonl_file(monkeypatch, mock_api, tmp_path):
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
    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
    ).mock(return_value=httpx2.Response(200, json=create_session_data))
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(return_value=httpx2.Response(200, json=session_detail_data))
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(return_value=httpx2.Response(200, json=messages_data))

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = DevinModel.Options(debug=True)

    list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=llm.Conversation(model=sut),
            key=API_KEY,
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


def test_no_debug_logging_when_debug_option_is_false(
    monkeypatch, mock_api, tmp_path
):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "exit",
                "status_detail": None,
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx2.Response(
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
    prompt.options = DevinModel.Options()

    list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=llm.Conversation(model=sut),
            key=API_KEY,
        )
    )

    log_dir = tmp_path / "devin"
    assert not log_dir.exists()


def test_debug_logging_preserves_non_ascii(monkeypatch, mock_api, tmp_path):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)

    japanese_message = "これはテストです"
    messages_data = {
        "items": [
            {
                "event_id": "evt-1",
                "source": "devin",
                "message": japanese_message,
                "created_at": 1000,
            },
        ],
        "end_cursor": None,
        "has_next_page": False,
    }
    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(return_value=httpx2.Response(200, json=messages_data))

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = DevinModel.Options(debug=True)

    list(
        sut.execute(
            prompt,
            stream=False,
            response=MagicMock(),
            conversation=llm.Conversation(model=sut),
            key=API_KEY,
        )
    )

    log_dir = tmp_path / "devin"
    jsonl_files = list(log_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1

    raw_content = jsonl_files[0].read_text(encoding="utf-8")
    assert japanese_message in raw_content
    assert "\\u" not in raw_content


def test_duplicate_messages_are_deduplicated(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        json__eq={"prompt": "Do work"},
    ).mock(
        return_value=httpx2.Response(
            status_code=200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )

    session_responses = [
        httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "working",
            },
        ),
        httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        ),
    ]
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(side_effect=session_responses)

    messages_poll1 = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-1",
                    "source": "devin",
                    "message": "Working on it",
                    "created_at": 1000,
                },
            ],
            "end_cursor": None,
            "has_next_page": False,
        },
    )
    messages_poll2 = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-1",
                    "source": "devin",
                    "message": "Working on it",
                    "created_at": 1000,
                },
                {
                    "event_id": "evt-2",
                    "source": "devin",
                    "message": "All done",
                    "created_at": 1001,
                },
            ],
            "end_cursor": None,
            "has_next_page": False,
        },
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(side_effect=[messages_poll1, messages_poll2])

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Do work"
    prompt.options = DevinModel.Options()

    with patch("llm_devin._devin.time.sleep"):
        actual = list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=llm.Conversation(model=sut),
                key=API_KEY,
            )
        )

    assert actual == ["Working on it", "\nAll done"]


def test_new_session_stores_session_id(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        json__eq={"prompt": "Hello"},
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "items": [
                    {
                        "event_id": "evt-1",
                        "source": "devin",
                        "message": "Hi!",
                        "created_at": 1000,
                    },
                ],
                "end_cursor": "cursor-1",
                "has_next_page": False,
            },
        )
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Hello"
    prompt.options = DevinModel.Options()
    response = MagicMock()

    list(
        sut.execute(
            prompt,
            stream=False,
            response=response,
            conversation=None,
            key=API_KEY,
        )
    )

    assert response.response_json == {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-1",
    }


def test_continue_conversation(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        json__eq={"message": "Follow up question"},
    ).mock(
        return_value=httpx2.Response(200, json={})
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        )
    )
    prefetch_response = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-1",
                    "source": "user",
                    "message": "Hello",
                    "created_at": 1000,
                },
                {
                    "event_id": "evt-2",
                    "source": "devin",
                    "message": "Previous answer",
                    "created_at": 1001,
                },
            ],
            "end_cursor": "cursor-1",
            "has_next_page": False,
        },
    )
    poll_response = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-1",
                    "source": "user",
                    "message": "Hello",
                    "created_at": 1000,
                },
                {
                    "event_id": "evt-2",
                    "source": "devin",
                    "message": "Previous answer",
                    "created_at": 1001,
                },
                {
                    "event_id": "evt-3",
                    "source": "devin",
                    "message": "Here is the follow-up answer.",
                    "created_at": 2000,
                },
            ],
            "end_cursor": "cursor-2",
            "has_next_page": False,
        },
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(side_effect=[prefetch_response, poll_response])

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Follow up question"
    prompt.options = DevinModel.Options()

    prev_response = MagicMock()
    prev_response.response_json = {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-1",
    }
    conversation = MagicMock()
    conversation.responses = [prev_response]

    response = MagicMock()

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=response,
            conversation=conversation,
            key=API_KEY,
        )
    )

    assert actual == ["Here is the follow-up answer."]
    assert response.response_json == {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-2",
    }


def test_continue_conversation_uses_previous_cursor(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        json__eq={"message": "Another question"},
    ).mock(
        return_value=httpx2.Response(200, json={})
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        )
    )
    prefetch_response = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-4",
                    "source": "devin",
                    "message": "Old message",
                    "created_at": 2500,
                },
            ],
            "end_cursor": "cursor-prev",
            "has_next_page": False,
        },
    )
    poll_response = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-4",
                    "source": "devin",
                    "message": "Old message",
                    "created_at": 2500,
                },
                {
                    "event_id": "evt-5",
                    "source": "devin",
                    "message": "Response after cursor",
                    "created_at": 3000,
                },
            ],
            "end_cursor": "cursor-new",
            "has_next_page": False,
        },
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        params__contains={"after": "cursor-prev"},
    ).mock(side_effect=[prefetch_response, poll_response])

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Another question"
    prompt.options = DevinModel.Options()

    prev_response = MagicMock()
    prev_response.response_json = {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-prev",
    }
    conversation = MagicMock()
    conversation.responses = [prev_response]

    response = MagicMock()

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=response,
            conversation=conversation,
            key=API_KEY,
        )
    )

    assert actual == ["Response after cursor"]
    assert response.response_json == {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-new",
    }


def test_continue_conversation_invalid_session_raises_model_error(
    monkeypatch, mock_api
):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-deleted-session/messages",
        params__contains={"after": "cursor-old"},
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "items": [],
                "end_cursor": "cursor-old",
                "has_next_page": False,
            },
        )
    )
    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-deleted-session/messages",
        json__eq={"message": "Follow up"},
    ).mock(
        return_value=httpx2.Response(404, json={"error": "session not found"})
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()

    prev_response = MagicMock()
    prev_response.response_json = {
        "session_id": "devin-deleted-session",
        "end_cursor": "cursor-old",
    }
    conversation = MagicMock()
    conversation.responses = [prev_response]

    with pytest.raises(
        llm.ModelError,
        match="The previous Devin session is invalid or expired",
    ):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=conversation,
                key=API_KEY,
            )
        )


def test_continue_conversation_server_error_is_reraised(
    monkeypatch, mock_api
):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        params__contains={"after": "cursor-old"},
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "items": [],
                "end_cursor": "cursor-old",
                "has_next_page": False,
            },
        )
    )
    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        json__eq={"message": "Follow up"},
    ).mock(
        return_value=httpx2.Response(500, json={"error": "internal server error"})
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()

    prev_response = MagicMock()
    prev_response.response_json = {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-old",
    }
    conversation = MagicMock()
    conversation.responses = [prev_response]

    with pytest.raises(httpx2.HTTPStatusError):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=conversation,
                key=API_KEY,
            )
        )


def test_continue_conversation_null_end_cursor_skips_old_messages(
    monkeypatch, mock_api
):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        json__eq={"message": "Follow up"},
    ).mock(
        return_value=httpx2.Response(200, json={})
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        )
    )
    prefetch_response = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-1",
                    "source": "user",
                    "message": "Hello",
                    "created_at": 1000,
                },
                {
                    "event_id": "evt-2",
                    "source": "devin",
                    "message": "Previous answer that should NOT appear again",
                    "created_at": 1001,
                },
            ],
            "end_cursor": None,
            "has_next_page": False,
        },
    )
    poll_response = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-1",
                    "source": "user",
                    "message": "Hello",
                    "created_at": 1000,
                },
                {
                    "event_id": "evt-2",
                    "source": "devin",
                    "message": "Previous answer that should NOT appear again",
                    "created_at": 1001,
                },
                {
                    "event_id": "evt-3",
                    "source": "user",
                    "message": "Follow up",
                    "created_at": 2000,
                },
                {
                    "event_id": "evt-4",
                    "source": "devin",
                    "message": "New follow-up answer",
                    "created_at": 2001,
                },
            ],
            "end_cursor": None,
            "has_next_page": False,
        },
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(side_effect=[prefetch_response, poll_response])

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()

    prev_response = MagicMock()
    prev_response.response_json = {
        "session_id": "devin-test-session",
        "end_cursor": None,
    }
    conversation = MagicMock()
    conversation.responses = [prev_response]

    response = MagicMock()

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=response,
            conversation=conversation,
            key=API_KEY,
        )
    )

    assert actual == ["New follow-up answer"]

    messages_gets = [
        c for c in mock_api.calls if c.method == "GET"
        and "/messages" in str(c.url)
    ]
    messages_post = [
        c for c in mock_api.calls if c.method == "POST"
        and "/messages" in str(c.url)
    ]
    assert len(messages_gets) >= 1
    assert len(messages_post) == 1
    first_get_index = len(mock_api.calls)
    post_index = len(mock_api.calls)
    for i, call in enumerate(mock_api.calls):
        url = str(call.url)
        if call.method == "GET" and "/messages" in url:
            first_get_index = min(first_get_index, i)
        if call.method == "POST" and "/messages" in url:
            post_index = min(post_index, i)
    assert first_get_index < post_index


def test_collect_existing_event_ids_pagination_error(monkeypatch, mock_api):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "items": [
                    {
                        "event_id": "evt-1",
                        "source": "devin",
                        "message": "Old",
                        "created_at": 1000,
                    },
                ],
                "end_cursor": None,
                "has_next_page": True,
            },
        )
    )

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()

    prev_response = MagicMock()
    prev_response.response_json = {
        "session_id": "devin-test-session",
        "end_cursor": None,
    }
    conversation = MagicMock()
    conversation.responses = [prev_response]

    with pytest.raises(
        llm.ModelError,
        match="messages pagination indicated another page without an end_cursor",
    ):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=conversation,
                key=API_KEY,
            )
        )


def test_devin_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        DevinModel.Options(devin_mode="turbo")


def _mock_first_session(mock_api, prompt_text="Hello"):
    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions",
        json__eq={"prompt": prompt_text},
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "url": "https://app.devin.ai/sessions/devin-test-session",
                "status": "running",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        )
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "items": [
                    {
                        "event_id": "evt-1",
                        "source": "devin",
                        "message": "First answer",
                        "created_at": 1000,
                    },
                ],
                "end_cursor": "cursor-1",
                "has_next_page": False,
            },
        )
    )


def _mock_follow_up(mock_api, message="Follow up"):
    mock_api.post(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
        json__eq={"message": message},
    ).mock(return_value=httpx2.Response(200, json={}))
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session",
    ).mock(
        return_value=httpx2.Response(
            200,
            json={
                "session_id": "devin-test-session",
                "status": "running",
                "status_detail": "finished",
            },
        )
    )
    prefetch_response = httpx2.Response(
        200,
        json={"items": [], "end_cursor": "cursor-1", "has_next_page": False},
    )
    poll_response = httpx2.Response(
        200,
        json={
            "items": [
                {
                    "event_id": "evt-3",
                    "source": "devin",
                    "message": "Follow-up answer",
                    "created_at": 2000,
                },
            ],
            "end_cursor": "cursor-2",
            "has_next_page": False,
        },
    )
    mock_api.get(
        f"{BASE_URL}/organizations/{ORG_ID}/sessions/devin-test-session/messages",
    ).mock(side_effect=[prefetch_response, poll_response])


def _log_first_turn(monkeypatch, mock_api, tmp_path):
    from sqlite_utils import Database
    from llm.migrations import migrate

    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)
    db_path = tmp_path / "logs.db"
    db = Database(db_path)
    migrate(db)

    _mock_first_session(mock_api)
    model = DevinModel()
    conversation = llm.Conversation(model=model)
    response = conversation.prompt("Hello", key=API_KEY, stream=False)
    assert response.text() == "First answer"
    response.log_to_db(db)
    assert response.response_json == {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-1",
    }
    return db_path, conversation.id


@pytest.mark.parametrize("use_cid", [True, False], ids=["--cid", "-c"])
def test_continue_conversation_loaded_from_llm_logs(
    monkeypatch, mock_api, tmp_path, use_cid
):
    from llm.cli import load_conversation

    db_path, conversation_id = _log_first_turn(monkeypatch, mock_api, tmp_path)

    loaded = load_conversation(
        conversation_id if use_cid else None, database=str(db_path)
    )
    assert loaded.id == conversation_id
    assert loaded.responses == []
    assert loaded.loaded_messages

    mock_api.routes.clear()
    mock_api.calls.clear()
    _mock_follow_up(mock_api)

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()
    response = MagicMock()

    actual = list(
        sut.execute(
            prompt,
            stream=False,
            response=response,
            conversation=loaded,
            key=API_KEY,
        )
    )

    assert actual == ["Follow-up answer"]
    assert response.response_json == {
        "session_id": "devin-test-session",
        "end_cursor": "cursor-2",
    }
    assert not any(
        c.method == "POST" and str(c.url).endswith("/sessions")
        for c in mock_api.calls
    )
    first_messages_get = next(
        c for c in mock_api.calls
        if c.method == "GET" and "/messages" in str(c.url)
    )
    assert first_messages_get.url.params.get("after") == "cursor-1"


def test_continue_conversation_without_session_id_does_not_create_session(
    monkeypatch, mock_api, tmp_path
):
    from llm.parts import Message, TextPart

    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setattr(llm, "user_dir", lambda: tmp_path)

    sut = DevinModel()
    conversation = llm.Conversation(model=sut)
    conversation.loaded_messages = [
        Message(role="user", parts=[TextPart(text="Hello")]),
        Message(role="assistant", parts=[TextPart(text="First answer")]),
    ]
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()

    with pytest.raises(llm.ModelError, match="Could not find the Devin session ID"):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=conversation,
                key=API_KEY,
            )
        )

    assert mock_api.calls == []


def test_legacy_responses_without_session_id_does_not_create_session(
    monkeypatch, mock_api
):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)

    sut = DevinModel()
    prev_response = MagicMock()
    prev_response.response_json = None
    conversation = MagicMock()
    conversation.id = None
    conversation.responses = [prev_response]
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()

    with pytest.raises(llm.ModelError, match="Could not find the Devin session ID"):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=conversation,
                key=API_KEY,
            )
        )

    assert mock_api.calls == []


def test_continue_after_other_model_turn_does_not_reuse_stale_session(
    monkeypatch, mock_api, tmp_path
):
    from llm.cli import load_conversation
    from sqlite_utils import Database

    db_path, conversation_id = _log_first_turn(monkeypatch, mock_api, tmp_path)

    other_model = llm.get_model("gpt-4o-mini")
    other_model.key = "dummy-openai-key"
    other_conversation = llm.Conversation(model=other_model, id=conversation_id)
    other_response = other_conversation.prompt("Hi", stream=False)
    with patch.object(other_model, "execute", return_value=iter(["Other"])):
        assert other_response.text() == "Other"
    other_response.log_to_db(Database(db_path))

    loaded = load_conversation(conversation_id, database=str(db_path))
    assert loaded.responses == []
    assert loaded.loaded_messages

    mock_api.routes.clear()
    mock_api.calls.clear()

    sut = DevinModel()
    prompt = MagicMock()
    prompt.prompt = "Follow up"
    prompt.options = DevinModel.Options()

    with pytest.raises(llm.ModelError, match="Could not find the Devin session ID"):
        list(
            sut.execute(
                prompt,
                stream=False,
                response=MagicMock(),
                conversation=loaded,
                key=API_KEY,
            )
        )

    assert mock_api.calls == []
