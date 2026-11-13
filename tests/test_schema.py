import json
from types import SimpleNamespace

import httpx2
import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli
from pydantic import ValidationError

from llm_devin import DevinModel


SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}
SESSION_ID = "devin-" + "a" * 32
URL = "https://app.devin.ai/sessions/" + "a" * 32
BASE = "https://api.devin.ai/v3/organizations/org-test/sessions"
SESSION = f"{BASE}/{SESSION_ID}"
MISSING = object()


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", "org-test")
    monkeypatch.setenv("LLM_DEVIN_KEY", "test-api-key")
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setattr("llm_devin._devin.time.sleep", lambda _: None)


def reply(data):
    return httpx2.Response(200, json=data)


def setup_session(api, value=MISSING, *, schema=SCHEMA, status="exit", detail=None):
    body = {"prompt": "Hello"}
    if schema is not None:
        body["structured_output_schema"] = schema
    api.post(BASE, json__eq=body).mock(return_value=reply({
        "session_id": SESSION_ID, "url": URL,
    }))
    data = {"status": status, "status_detail": detail}
    if value is not MISSING:
        data["structured_output"] = value
    api.get(SESSION).mock(return_value=reply(data))
    api.get(SESSION + "/messages").mock(return_value=reply({
        "items": [{"event_id": "evt-1", "source": "devin", "message": "Chat"}],
        "end_cursor": "cursor-1", "has_next_page": False,
    }))


def execute(*, schema=SCHEMA, conversation=None, **options):
    model = DevinModel()
    response = SimpleNamespace(response_json=None)
    prompt = SimpleNamespace(prompt="Hello", schema=schema, options=model.Options(**options))
    chunks = list(model.execute(prompt, False, response, conversation, "test-api-key"))
    return chunks, response.response_json


@pytest.mark.parametrize("value", [{"name": "日本語"}, {}, [], False, 0])
def test_schema_output(mock_api, capsys, value):
    setup_session(mock_api, value)
    chunks, metadata = execute(debug=True)
    assert chunks == [json.dumps(value, ensure_ascii=False)]
    print("".join(chunks))
    captured = capsys.readouterr()
    assert json.loads(captured.out) == value
    assert "Devin URL:" in captured.err
    assert "Debug log:" in captured.err
    assert "Chat" not in captured.out
    assert metadata == {
        "session_id": SESSION_ID, "end_cursor": "cursor-1",
        "structured_output_schema": SCHEMA, "structured_output": value,
    }


@pytest.mark.parametrize("value", [MISSING, None])
def test_missing_output(mock_api, value):
    setup_session(mock_api, value, status="running", detail="finished")
    with pytest.raises(llm.ModelError, match="status=running.*status_detail=finished.*without structured output") as exc:
        execute()
    assert URL in str(exc.value)


def test_error_status(mock_api):
    setup_session(mock_api, {"name": "Ignored"}, status="error")
    with pytest.raises(llm.ModelError, match="status=error"):
        execute()


def history(schema=SCHEMA):
    return SimpleNamespace(responses=[SimpleNamespace(response_json={
        "session_id": SESSION_ID, "end_cursor": "old-cursor",
        "structured_output_schema": schema,
    })])


@pytest.mark.parametrize("recorded", [None, {}, {"type": "array"}])
def test_changed_schema_rejected_without_http(mock_api, recorded):
    with pytest.raises(llm.ModelError, match="cannot change.*schema"):
        execute(conversation=history(recorded))
    assert mock_api.calls == []


def test_explicit_session_without_history_rejected(mock_api, capsys):
    with pytest.raises(llm.ModelError, match="cannot change.*schema"):
        execute(session=URL)
    assert mock_api.calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Continuing Devin session:" in captured.err


def setup_continuation(api, final, *, schema_mode=True):
    details = [{"status": "exit", "structured_output": {"name": "old"}},
               {"status": "exit", "structured_output": final}]
    api.get(SESSION).mock(side_effect=[reply(d) for d in details] if schema_mode
                          else [reply(details[-1])])
    api.get(SESSION + "/messages").mock(side_effect=[
        reply({"items": [], "end_cursor": "old-cursor"}),
        reply({"items": [{"event_id": "new", "source": "devin", "message": "New chat"}],
               "end_cursor": "new-cursor"}),
    ])
    api.post(SESSION + "/messages", json__eq={"message": "Hello"}).mock(return_value=reply({}))


def test_same_schema_continuation(mock_api):
    setup_continuation(mock_api, {"name": "new"})
    chunks, metadata = execute(conversation=history(dict(SCHEMA)))
    assert json.loads(chunks[0]) == {"name": "new"}
    assert metadata["end_cursor"] == "new-cursor"
    assert [(r.method, str(r.url).split("?")[0]) for r in mock_api.calls] == [
        ("GET", SESSION), ("GET", SESSION + "/messages"),
        ("POST", SESSION + "/messages"), ("GET", SESSION),
        ("GET", SESSION + "/messages"),
    ]


@pytest.mark.parametrize("final", [{"name": "old"}, None])
def test_stale_continuation(mock_api, final):
    setup_continuation(mock_api, final)
    with pytest.raises(llm.ModelError, match=f"No new structured output.*llm devin status {URL}"):
        execute(conversation=history())


def test_continuation_without_schema(mock_api):
    setup_continuation(mock_api, None, schema_mode=False)
    chunks, metadata = execute(schema=None, conversation=history())
    assert chunks == ["New chat"]
    assert metadata == {
        "session_id": SESSION_ID, "end_cursor": "new-cursor",
        "structured_output_schema": SCHEMA,
    }


def test_schema_continuation_after_plain_text_turn(mock_api):
    setup_continuation(mock_api, None, schema_mode=False)
    _, metadata = execute(schema=None, conversation=history())
    conversation = SimpleNamespace(responses=[SimpleNamespace(response_json=metadata)])
    mock_api.routes.clear()
    setup_continuation(mock_api, {"name": "new"})
    chunks, _ = execute(conversation=conversation)
    assert chunks == ['{"name": "new"}']


@pytest.mark.parametrize("mode", ["fast-ultra", "swe-2-medium", "swe-2-high", "swe-2-max"])
def test_new_modes(mode):
    assert DevinModel.Options(devin_mode=mode).devin_mode == mode


def test_unknown_mode():
    with pytest.raises(ValidationError):
        DevinModel.Options(devin_mode="unknown")


def test_schema_cli(mock_api):
    schema = {"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"]}
    setup_session(mock_api, {"name": "Ada", "age": 36}, schema=schema)
    result = CliRunner().invoke(cli, ["-m", "devin", "--schema", "name, age int", "Hello"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"name": "Ada", "age": 36}
    assert "Devin URL:" in result.stderr


@pytest.mark.parametrize("schema", [None, SCHEMA])
def test_state_preserves_schema(schema):
    state = DevinModel._state_from_response_json({
        "session_id": SESSION_ID, "structured_output_schema": schema,
    })
    assert state["structured_output_schema"] == schema


def test_schema_continuation_from_logs(mock_api, tmp_path):
    from llm.cli import load_conversation
    from llm.migrations import migrate
    from sqlite_utils import Database

    setup_session(mock_api, {"name": "first"})
    model = DevinModel()
    conversation = llm.Conversation(model=model)
    first = conversation.prompt("Hello", schema=SCHEMA, key="test-api-key", stream=False)
    assert json.loads(first.text()) == {"name": "first"}
    db = Database(tmp_path / "logs.db")
    migrate(db)
    first.log_to_db(db)
    loaded = load_conversation(conversation.id, database=str(tmp_path / "logs.db"))
    state = model._get_previous_state(loaded, None, schema_mode=True)
    assert state["structured_output_schema"] == SCHEMA
