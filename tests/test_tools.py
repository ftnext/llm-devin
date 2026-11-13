import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx2
import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli
from llm.plugins import pm

from llm_devin import Devin, _devin

API_KEY = "test-api-key"
ORG_ID = "org-test123"
BASE_URL = "https://api.devin.ai/v3"
SESSION_HEX = "0123456789abcdef0123456789abcdef"
SESSION_ID = f"devin-{SESSION_HEX}"
SESSION_URL = f"https://app.devin.ai/sessions/{SESSION_HEX}"

SESSIONS_ENDPOINT = f"{BASE_URL}/organizations/{ORG_ID}/sessions"
SESSION_ENDPOINT = f"{SESSIONS_ENDPOINT}/{SESSION_ID}"
MESSAGES_ENDPOINT = f"{SESSION_ENDPOINT}/messages"


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", ORG_ID)
    monkeypatch.setenv("LLM_DEVIN_KEY", API_KEY)


def created(**overrides):
    payload = {
        "session_id": SESSION_ID,
        "url": SESSION_URL,
        "status": "new",
        "acus_consumed": 0,
        "created_at": 1000,
        "updated_at": 1000,
        "pull_requests": [],
        "tags": [],
    }
    payload.update(overrides)
    return httpx2.Response(200, json=payload)


def message(event_id, source, text):
    return {
        "event_id": event_id,
        "source": source,
        "message": text,
        "created_at": 1000,
    }


class TestRegistration:
    def test_toolbox_is_registered(self):
        tools = llm.get_tools()

        assert tools["Devin"] is Devin
        assert {tool.name for tool in Devin().tools()} == {
            "Devin_start_session",
            "Devin_session_status",
            "Devin_session_messages",
        }

    def test_tools_list_cli(self):
        result = CliRunner().invoke(cli, ["tools", "list", "--json"])

        assert result.exit_code == 0, result.output
        toolboxes = json.loads(result.output)["toolboxes"]
        devin = next(box for box in toolboxes if box["name"] == "Devin")
        assert {tool["name"] for tool in devin["tools"]} == {
            "Devin_start_session",
            "Devin_session_status",
            "Devin_session_messages",
        }


class TestConfiguration:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_acu_limit": 0},
            {"max_acu_limit": -1},
            {"max_acu_limit": 51},
            {"max_acu_limit": True},
            {"max_acu_limit": "5"},
            {"max_sessions": 0},
            {"max_sessions": 11},
            {"devin_mode": "ultra"},
            {"devin_mode": "fast"},
        ],
    )
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError, match="Devin toolbox"):
            Devin(**kwargs)


class TestStartSession:
    def test_defaults_set_bounded_acu_limit_and_return_immediately(
        self, mock_api
    ):
        mock_api.post(
            SESSIONS_ENDPOINT,
            json__eq={"prompt": "Fix the flaky test", "max_acu_limit": 5},
        ).mock(return_value=created())

        result = Devin().start_session("  Fix the flaky test  ")

        assert result["session_id"] == SESSION_ID
        assert result["url"] == SESSION_URL
        assert result["status"] == "new"
        assert result["max_acu_limit"] == 5
        assert "Devin_session_status" in result["note"]
        assert [call.method for call in mock_api.calls] == ["POST"]

    @pytest.mark.parametrize(
        "mode", ["normal", "lite", "fusion", "swe-2-medium", "swe-2-high", "swe-2-max"]
    )
    def test_supported_modes(self, mock_api, mode):
        mock_api.post(
            SESSIONS_ENDPOINT,
            json__eq={
                "prompt": "Task",
                "max_acu_limit": 2,
                "devin_mode": mode,
                "repos": ["owner/repo", "owner/other.repo"],
                "title": "Delegated task",
            },
        ).mock(return_value=created(devin_mode=mode))

        result = Devin(max_acu_limit=3).start_session(
            "Task",
            devin_mode=mode,
            repos="owner/repo, owner/other.repo",
            title="Delegated task",
            max_acu_limit=2,
        )

        assert result["devin_mode"] == mode
        assert result["max_acu_limit"] == 2

    def test_configured_mode_is_used(self, mock_api):
        mock_api.post(
            SESSIONS_ENDPOINT,
            json__eq={
                "prompt": "Task",
                "max_acu_limit": 5,
                "devin_mode": "fusion",
            },
        ).mock(return_value=created(devin_mode="fusion"))

        result = Devin(devin_mode="fusion").start_session("Task")

        assert result["devin_mode"] == "fusion"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"prompt": ""}, "prompt"),
            ({"prompt": "   "}, "prompt"),
            pytest.param({"prompt": "x" * 30001}, "prompt", id="prompt-too-long"),
            ({"prompt": "Task", "devin_mode": "ultra"}, "devin_mode"),
            ({"prompt": "Task", "devin_mode": "fast"}, "devin_mode"),
            ({"prompt": "Task", "devin_mode": ""}, "devin_mode"),
            ({"prompt": "Task", "max_acu_limit": 6}, "between 1 and 5"),
            ({"prompt": "Task", "max_acu_limit": 0}, "between 1 and 5"),
            ({"prompt": "Task", "repos": "https://github.com/o/r"}, "owner/repo"),
        ],
    )
    def test_invalid_input_sends_no_request(self, mock_api, kwargs, match):
        with pytest.raises(Exception, match=match):
            Devin().start_session(**kwargs)

        assert mock_api.calls == []

    def test_configured_mode_cannot_be_overridden(self, mock_api):
        with pytest.raises(Exception, match="fixed to 'fusion'"):
            Devin(devin_mode="fusion").start_session(
                "Task", devin_mode="swe-2-high"
            )

        assert mock_api.calls == []

    def test_session_budget_blocks_second_start(self, mock_api):
        mock_api.post(SESSIONS_ENDPOINT).mock(return_value=created())
        toolbox = Devin()
        toolbox.start_session("Task")

        with pytest.raises(Exception, match=f"at most 1 session.*{SESSION_ID}"):
            toolbox.start_session("Task again")

        assert len(mock_api.calls) == 1

    def test_concurrent_starts_respect_budget(self, monkeypatch, mock_api):
        mock_api.post(SESSIONS_ENDPOINT).mock(return_value=created())
        both_connecting = threading.Barrier(2, timeout=5)
        create_http_client = _devin.create_http_client

        def synchronized_client(headers):
            both_connecting.wait()
            return create_http_client(headers)

        monkeypatch.setattr(_devin, "create_http_client", synchronized_client)
        toolbox = Devin()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(toolbox.start_session, "Task") for _ in range(2)]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=10)["session_id"])
                except Exception as ex:
                    outcomes.append(str(ex))

        assert len(mock_api.calls) == 1
        assert SESSION_ID in outcomes
        assert any("at most 1 session" in outcome for outcome in outcomes)

    def test_client_error_does_not_use_budget(self, mock_api):
        mock_api.post(SESSIONS_ENDPOINT).mock(
            side_effect=[
                httpx2.Response(422, json={"detail": "invalid repos"}),
                created(),
            ]
        )
        toolbox = Devin()

        with pytest.raises(Exception, match=r"HTTP 422 \(invalid repos\)"):
            toolbox.start_session("Task")
        result = toolbox.start_session("Task")

        assert result["session_id"] == SESSION_ID

    @pytest.mark.parametrize(
        "failure, reason",
        [
            (httpx2.Response(500), "HTTP 500"),
            (
                httpx2.Response(502, json={"detail": "bad gateway"}),
                "HTTP 502 (bad gateway)",
            ),
            (httpx2.ReadTimeout("timed out"), "ReadTimeout"),
            (httpx2.WriteTimeout("timed out"), "WriteTimeout"),
            (httpx2.RemoteProtocolError("closed"), "RemoteProtocolError"),
            (httpx2.ReadError("reset"), "ReadError"),
            (httpx2.Response(200, text="<html>"), "unexpected response"),
            (httpx2.Response(200, json={"status": "new"}), "unexpected response"),
        ],
    )
    def test_uncertain_start_warns_session_may_exist(
        self, mock_api, failure, reason
    ):
        route = mock_api.post(SESSIONS_ENDPOINT)
        if isinstance(failure, Exception):
            route.mock(side_effect=[failure])
        else:
            route.mock(return_value=failure)
        toolbox = Devin()

        with pytest.raises(Exception) as excinfo:
            toolbox.start_session("Task")

        assert str(excinfo.value) == (
            f"Could not confirm whether a Devin session was started ({reason})."
            " A session may already exist and consume ACUs: check the Devin web"
            " app before retrying. This toolbox will not start another session."
        )
        assert excinfo.value.__cause__ is None
        with pytest.raises(Exception, match="at most 1 session"):
            toolbox.start_session("Task")
        assert len(mock_api.calls) == 1

    @pytest.mark.parametrize(
        "failure",
        [httpx2.ConnectError("refused"), httpx2.ConnectTimeout("timed out")],
    )
    def test_connect_failure_is_not_reported_as_maybe_created(
        self, mock_api, failure
    ):
        mock_api.post(SESSIONS_ENDPOINT).mock(side_effect=[failure])
        toolbox = Devin()

        with pytest.raises(Exception) as excinfo:
            toolbox.start_session("Task")

        assert str(excinfo.value) == (
            "Could not start a Devin session: could not reach the Devin API"
            f" ({type(failure).__name__})."
        )
        with pytest.raises(Exception, match="at most 1 session"):
            toolbox.start_session("Task")
        assert len(mock_api.calls) == 1

    @pytest.mark.parametrize("status_code", [401, 403, 422])
    def test_rejection_is_not_reported_as_maybe_created(
        self, mock_api, status_code
    ):
        mock_api.post(SESSIONS_ENDPOINT).mock(
            return_value=httpx2.Response(status_code)
        )

        with pytest.raises(Exception) as excinfo:
            Devin().start_session("Task")

        assert str(excinfo.value).startswith(
            f"Could not start a Devin session: Devin API returned HTTP {status_code}."
        )
        assert "may already exist" not in str(excinfo.value)

    def test_missing_key(self, monkeypatch, mock_api):
        monkeypatch.delenv("LLM_DEVIN_KEY")

        with pytest.raises(Exception, match="No Devin API key found"):
            Devin().start_session("Task")

        assert mock_api.calls == []

    def test_missing_org_id(self, monkeypatch, mock_api):
        monkeypatch.delenv("LLM_DEVIN_ORG_ID")

        with pytest.raises(Exception, match="LLM_DEVIN_ORG_ID"):
            Devin().start_session("Task")

        assert mock_api.calls == []


class TestErrorSanitization:
    def test_auth_error_does_not_leak_key(self, mock_api):
        mock_api.post(SESSIONS_ENDPOINT).mock(
            return_value=httpx2.Response(401, json={"detail": "bad token"})
        )

        with pytest.raises(Exception) as excinfo:
            Devin().start_session("Task")

        assert "HTTP 401 (bad token)" in str(excinfo.value)
        assert API_KEY not in str(excinfo.value)
        assert excinfo.value.__cause__ is None

    def test_network_error_does_not_leak_request(self, mock_api):
        mock_api.get(SESSION_ENDPOINT).mock(
            side_effect=[httpx2.ConnectError(f"Bearer {API_KEY} {ORG_ID}")]
        )

        with pytest.raises(Exception) as excinfo:
            Devin().session_status(SESSION_ID)

        assert str(excinfo.value) == (
            "Could not read the Devin session:"
            " could not reach the Devin API (ConnectError)."
        )

    def test_long_detail_is_truncated(self, mock_api):
        mock_api.get(SESSION_ENDPOINT).mock(
            return_value=httpx2.Response(500, json={"detail": "x" * 1000})
        )

        with pytest.raises(Exception) as excinfo:
            Devin().session_status(SESSION_ID)

        assert len(str(excinfo.value)) < 300


class TestSessionStatus:
    def test_running_session(self, mock_api):
        mock_api.get(SESSION_ENDPOINT).mock(
            return_value=created(
                status="running",
                status_detail="working",
                title="Fix flaky test",
                devin_mode="fusion",
                acus_consumed=1.5,
                updated_at=2000,
                pull_requests=[
                    {"pr_url": "https://github.com/o/r/pull/1", "pr_state": "open"}
                ],
                structured_output={"result": "ok"},
            )
        )

        result = Devin().session_status(SESSION_URL)

        assert result == {
            "session_id": SESSION_ID,
            "url": SESSION_URL,
            "status": "running",
            "status_detail": "working",
            "title": "Fix flaky test",
            "devin_mode": "fusion",
            "acus_consumed": 1.5,
            "updated_at": 2000,
            "pull_requests": [
                {"url": "https://github.com/o/r/pull/1", "state": "open"}
            ],
            "structured_output": {"result": "ok"},
            "note": (
                "Devin is still working asynchronously. Do not poll in a loop:"
                " share the session URL with the user and check again later."
            ),
        }
        assert [call.method for call in mock_api.calls] == ["GET"]

    @pytest.mark.parametrize(
        "status, status_detail",
        [
            ("running", "finished"),
            ("running", "waiting_for_user"),
            ("exit", None),
            ("suspended", "inactivity"),
        ],
    )
    def test_settled_session_has_no_note(self, mock_api, status, status_detail):
        mock_api.get(SESSION_ENDPOINT).mock(
            return_value=created(status=status, status_detail=status_detail)
        )

        result = Devin().session_status(SESSION_ID)

        assert "note" not in result

    def test_invalid_reference_sends_no_request(self, mock_api):
        with pytest.raises(Exception, match="Invalid Devin session"):
            Devin().session_status("not-a-session")

        assert mock_api.calls == []

    def test_inaccessible_session(self, mock_api):
        mock_api.get(SESSION_ENDPOINT).mock(
            return_value=httpx2.Response(404, json={"detail": "Not found"})
        )

        with pytest.raises(Exception, match="may not be accessible"):
            Devin().session_status(SESSION_ID)

        assert [call.method for call in mock_api.calls] == ["GET"]


class TestSessionMessages:
    def test_paginates_and_returns_latest_devin_messages(self, mock_api):
        mock_api.get(MESSAGES_ENDPOINT, params__contains={"after": "c1"}).mock(
            return_value=httpx2.Response(
                200,
                json={
                    "items": [
                        message("e3", "devin", "Working on it"),
                        message("e4", "devin", "Done: opened PR"),
                    ],
                    "end_cursor": None,
                    "has_next_page": False,
                },
            )
        )
        mock_api.get(MESSAGES_ENDPOINT, params__contains={"first": "200"}).mock(
            return_value=httpx2.Response(
                200,
                json={
                    "items": [
                        message("e1", "user", "Fix it"),
                        message("e2", "devin", "Looking"),
                    ],
                    "end_cursor": "c1",
                    "has_next_page": True,
                },
            )
        )

        result = Devin().session_messages(SESSION_ID, limit=2)

        assert result == {
            "session_id": SESSION_ID,
            "url": SESSION_URL,
            "messages": [
                {"source": "devin", "created_at": 1000, "message": "Working on it"},
                {"source": "devin", "created_at": 1000, "message": "Done: opened PR"},
            ],
        }
        assert all(call.method == "GET" for call in mock_api.calls)

    def test_all_sources_and_truncation(self, mock_api):
        mock_api.get(MESSAGES_ENDPOINT).mock(
            return_value=httpx2.Response(
                200,
                json={
                    "items": [
                        message("e1", "user", "Fix it"),
                        message("e2", "devin", "y" * 5000),
                    ],
                    "has_next_page": False,
                },
            )
        )

        result = Devin().session_messages(SESSION_ID, source="all")

        first, second = result["messages"]
        assert first["source"] == "user"
        assert len(second["message"]) == 4000
        assert second["truncated"] is True

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"limit": 0}, "limit"),
            ({"limit": 21}, "limit"),
            ({"source": "system"}, "source"),
        ],
    )
    def test_invalid_input_sends_no_request(self, mock_api, kwargs, match):
        with pytest.raises(Exception, match=match):
            Devin().session_messages(SESSION_ID, **kwargs)

        assert mock_api.calls == []

    def test_next_page_without_cursor(self, mock_api):
        mock_api.get(MESSAGES_ENDPOINT).mock(
            return_value=httpx2.Response(
                200, json={"items": [], "has_next_page": True}
            )
        )

        with pytest.raises(Exception, match="without an end_cursor"):
            Devin().session_messages(SESSION_ID)


class ToolCallingModel(llm.Model):
    model_id = "test-tool-caller"
    supports_tools = True

    def execute(self, prompt, stream, response, conversation):
        if prompt.tool_results:
            yield "Delegated: " + prompt.tool_results[0].output
            return
        for tool_call in json.loads(prompt.prompt):
            response.add_tool_call(llm.ToolCall(**tool_call))
        yield ""


class ToolCallingPlugin:
    @llm.hookimpl
    def register_models(self, register):
        register(ToolCallingModel())


@pytest.fixture
def tool_calling_model():
    plugin = ToolCallingPlugin()
    pm.register(plugin, name="test-tool-caller")
    try:
        yield
    finally:
        pm.unregister(name="test-tool-caller")


class TestCli:
    def test_model_delegates_with_configured_toolbox(
        self, mock_api, tool_calling_model
    ):
        mock_api.post(
            SESSIONS_ENDPOINT,
            json__eq={
                "prompt": "Fix the flaky test",
                "max_acu_limit": 3,
                "devin_mode": "swe-2-high",
            },
        ).mock(return_value=created(devin_mode="swe-2-high"))
        tool_calls = [
            {
                "name": "Devin_start_session",
                "arguments": {"prompt": "Fix the flaky test"},
            }
        ]

        result = CliRunner().invoke(
            cli,
            [
                "-m",
                "test-tool-caller",
                "-T",
                'Devin(max_acu_limit=3, devin_mode="swe-2-high")',
                "--td",
                json.dumps(tool_calls),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Tool call: Devin_start_session" in result.output
        assert SESSION_URL in result.output
        assert '"max_acu_limit": 3' in result.output
        assert API_KEY not in result.output

    def test_tool_error_is_returned_to_model(
        self, mock_api, tool_calling_model
    ):
        tool_calls = [
            {
                "name": "Devin_start_session",
                "arguments": {"prompt": "Task", "devin_mode": "ultra"},
            }
        ]

        result = CliRunner().invoke(
            cli,
            ["-m", "test-tool-caller", "-T", "Devin", json.dumps(tool_calls)],
        )

        assert result.exit_code == 0, result.output
        assert "Delegated: Error: devin_mode must be one of" in result.output
        assert mock_api.calls == []
