from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Optional
from urllib.parse import urlparse

import httpx2
import llm
import sqlite_utils
from pydantic import Field
from pythonjsonlogger.json import JsonFormatter

try:
    from llm.logs import LogStore
except ImportError:
    LogStore = None


logger = logging.getLogger(__name__)

TIMEOUT = httpx2.Timeout(5.0, read=10.0)

SESSION_URL_BASE = "https://app.devin.ai/sessions/"
SESSION_ID_PATTERN = re.compile(r"^(?:devin-)?([0-9a-fA-F]{32})$")
API_BASE_URL = "https://api.devin.ai/v3"
ORG_ID_ENV_VAR = "LLM_DEVIN_ORG_ID"


def resolve_org_id() -> str:
    org_id = os.environ.get(ORG_ID_ENV_VAR, "")
    if not org_id:
        raise llm.ModelError(
            f"{ORG_ID_ENV_VAR} environment variable is required"
        )
    return org_id


def parse_session_reference(value: str) -> str:
    value = value.strip()
    candidate = value
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname != "app.devin.ai":
            raise llm.ModelError(
                f"Invalid Devin session URL: {value!r}"
                f" (expected {SESSION_URL_BASE}<id>)"
            )
        segments = [s for s in parsed.path.split("/") if s]
        if len(segments) != 2 or segments[0] != "sessions":
            raise llm.ModelError(
                f"Invalid Devin session URL: {value!r}"
                f" (expected {SESSION_URL_BASE}<id>)"
            )
        candidate = segments[1]
    match = SESSION_ID_PATTERN.match(candidate)
    if match is None:
        raise llm.ModelError(
            f"Invalid Devin session ID or URL: {value!r}"
            " (expected devin-<32 hex chars>, <32 hex chars>, or"
            f" {SESSION_URL_BASE}<32 hex chars>)"
        )
    return f"devin-{match.group(1).lower()}"


def session_url(session_id: str) -> str:
    return SESSION_URL_BASE + session_id.removeprefix("devin-")


def create_http_client(headers: dict[str, str]) -> httpx2.Client:
    return httpx2.Client(headers=headers, timeout=TIMEOUT)


def print_immediately(*objects) -> None:
    # ref: https://github.com/simonw/llm/blob/0.26/llm/cli.py#L867-L868
    print(*objects)
    sys.stdout.flush()


class DevinModel(llm.KeyModel):
    needs_key = "devin"
    key_env_var = "LLM_DEVIN_KEY"
    can_stream = True

    BASE_URL = API_BASE_URL

    class Options(llm.Options):
        debug: Optional[bool] = Field(
            description="Enable debug logging of API responses to JSONL file",
            default=False,
        )
        title: Optional[str] = Field(
            description="Custom title for the new session",
            default=None,
        )
        tags: Optional[str] = Field(
            description="Comma-separated tags to add to the new session",
            default=None,
        )
        repos: Optional[str] = Field(
            description="Comma-separated repositories (owner/repo) for the new session",
            default=None,
        )
        max_acu_limit: Optional[int] = Field(
            description="Maximum ACU limit for the new session",
            default=None,
        )
        playbook_id: Optional[str] = Field(
            description="Playbook ID to use for the new session",
            default=None,
        )
        devin_mode: Optional[str] = Field(
            description="Devin agent mode for the new session"
            " (normal, fast, lite, ultra, fusion)",
            default=None,
            pattern="^(normal|fast|lite|ultra|fusion)$",
        )
        session: Optional[str] = Field(
            description="Existing Devin session ID or URL to send the message to"
            " (cannot be combined with -c/--cid on a conversation with history)",
            default=None,
        )

    def __init__(self) -> None:
        self.model_id = "devin"

    def _org_id(self) -> str:
        return resolve_org_id()

    def _setup_debug_logging(self, debug: bool) -> logging.FileHandler | None:
        if not debug:
            return None
        log_dir = llm.user_dir() / "devin"
        log_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        log_file = log_dir / f"{timestamp}.jsonl"
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            JsonFormatter(timestamp=True, json_ensure_ascii=False)
        )
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        print_immediately(f"Debug log: {log_file}")
        return handler

    def _teardown_debug_logging(
        self, handler: logging.FileHandler | None
    ) -> None:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
            logger.setLevel(logging.NOTSET)

    def execute(self, prompt, stream, response, conversation, key):
        debug = prompt.options.debug or False
        handler = self._setup_debug_logging(debug)
        try:
            yield from self._execute(
                prompt, stream, response, conversation, key
            )
        finally:
            self._teardown_debug_logging(handler)

    @staticmethod
    def _has_history(conversation) -> bool:
        if conversation is None:
            return False
        if getattr(conversation, "responses", None):
            return True
        if getattr(conversation, "loaded_messages", None):
            return True
        return False

    @staticmethod
    def _state_from_response_json(response_json) -> dict | None:
        if not isinstance(response_json, dict):
            return None
        session_id = response_json.get("session_id")
        if not session_id:
            return None
        return {
            "session_id": session_id,
            "end_cursor": response_json.get("end_cursor"),
        }

    def _state_from_responses(self, conversation) -> dict | None:
        responses = getattr(conversation, "responses", None)
        if not responses:
            return None
        return self._state_from_response_json(responses[-1].response_json)

    def _state_from_logs_db(self, conversation) -> dict | None:
        if LogStore is None:
            return None
        conversation_id = getattr(conversation, "id", None)
        if not conversation_id:
            return None
        db_path = llm.user_dir() / "logs.db"
        if not db_path.exists():
            return None
        db = sqlite_utils.Database(db_path)
        if "turns" not in db.table_names():
            return None
        rows = db.query(
            "select id, model from turns where thread_id = ?"
            " order by id desc limit 1",
            [conversation_id],
        )
        row = next(iter(rows), None)
        if row is None or row["model"] != self.model_id:
            return None
        response_json = LogStore(db).turn_response_json(row["id"])
        return self._state_from_response_json(response_json)

    def _get_previous_state(self, conversation, explicit_session) -> dict | None:
        if explicit_session is not None:
            session_id = parse_session_reference(explicit_session)
            if self._has_history(conversation):
                state = self._state_from_responses(conversation)
                if state is None:
                    state = self._state_from_logs_db(conversation)
                if state is None or state["session_id"] != session_id:
                    raise llm.ModelError(
                        "The session option cannot be combined with -c/--cid"
                        " on a conversation that already has history."
                        " Omit -c/--cid to send to the specified session,"
                        " or omit the session option to continue the conversation."
                    )
            else:
                state = {"session_id": session_id, "end_cursor": None}
            print_immediately("Continuing Devin session:", session_url(session_id))
            return state
        if not self._has_history(conversation):
            return None
        state = self._state_from_responses(conversation)
        if state is None:
            state = self._state_from_logs_db(conversation)
        if state is None:
            raise llm.ModelError(
                "Could not find the Devin session ID for this conversation."
                " Please start a new conversation."
            )
        return state

    @staticmethod
    def _explicit_session_error(session_id, ex) -> llm.ModelError:
        if isinstance(ex, httpx2.HTTPStatusError):
            reason = f"HTTP {ex.response.status_code}"
        else:
            reason = f"{type(ex).__name__}: {ex}"
        return llm.ModelError(
            f"Could not reach Devin session {session_id}"
            f" ({session_url(session_id)}): {reason}."
            " The session may not exist, may have expired,"
            " or may not be accessible with this API key."
        )

    def _execute(self, prompt, stream, response, conversation, key):
        org_id = self._org_id()
        headers = {"Authorization": f"Bearer {key}"}
        with create_http_client(headers) as client:
            yield from self._run(client, prompt, response, conversation, org_id)

    def _run(self, client, prompt, response, conversation, org_id):
        previous_state = self._get_previous_state(
            conversation, prompt.options.session
        )

        seen_event_ids: set[str] = set()

        if previous_state is not None:
            session_id = previous_state["session_id"]
            logger.debug(
                "Continuing session %s", session_id,
            )

            poll_state: dict = {"cursor": previous_state["end_cursor"]}
            explicit = prompt.options.session is not None

            try:
                self._collect_existing_event_ids(
                    client, org_id, session_id, poll_state, seen_event_ids,
                )
            except (httpx2.RequestError, httpx2.HTTPStatusError) as ex:
                if explicit:
                    raise self._explicit_session_error(session_id, ex) from ex

            try:
                send_message_response = client.post(
                    f"{self.BASE_URL}/organizations/{org_id}/sessions/{session_id}/messages",
                    json={"message": prompt.prompt},
                    )
                send_message_response.raise_for_status()
            except httpx2.HTTPStatusError as ex:
                status_code = ex.response.status_code
                if explicit and status_code in {401, 403, 404, 410}:
                    raise self._explicit_session_error(session_id, ex) from ex
                if status_code in {404, 410}:
                    raise llm.ModelError(
                        "The previous Devin session is invalid or expired. "
                        "Please start a new conversation."
                    ) from ex
                raise
        else:
            request_json = self._build_create_session_json(prompt)
            logger.debug("Request JSON: %s", request_json)
            create_session_response = client.post(
                f"{self.BASE_URL}/organizations/{org_id}/sessions",
                json=request_json,
            )
            create_session_response.raise_for_status()

            create_session_data = create_session_response.json()
            logger.debug(
                "create_session response",
                extra={"data": create_session_data},
            )
            session_id = create_session_data["session_id"]
            print_immediately("Devin URL:", create_session_data["url"])
            poll_state = {"cursor": None}

        devin_messages: list[str] = []
        while True:
            try:
                session_detail = self._get_session(client, org_id, session_id)
            except (httpx2.RequestError, httpx2.HTTPStatusError):
                pass
            else:
                try:
                    yield from self._drain_messages(
                        client, org_id, session_id,
                        devin_messages, poll_state,
                        seen_event_ids,
                    )
                except (httpx2.RequestError, httpx2.HTTPStatusError):
                    pass

                status = session_detail["status"]
                status_detail = session_detail.get("status_detail")
                if status in {"exit", "error", "suspended"}:
                    break
                if status == "running" and status_detail in {
                    "finished",
                    "waiting_for_user",
                    "waiting_for_approval",
                }:
                    break
            time.sleep(5)

        response.response_json = {
            "session_id": session_id,
            "end_cursor": poll_state["cursor"],
        }

    @staticmethod
    def _split_csv(value: str | None) -> list[str] | None:
        if value is None:
            return None
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None

    def _build_create_session_json(self, prompt) -> dict:
        options = prompt.options
        request_json: dict = {"prompt": prompt.prompt}
        optional_fields = {
            "title": options.title,
            "tags": self._split_csv(options.tags),
            "repos": self._split_csv(options.repos),
            "max_acu_limit": options.max_acu_limit,
            "playbook_id": options.playbook_id,
            "devin_mode": options.devin_mode,
        }
        for name, value in optional_fields.items():
            if value is not None:
                request_json[name] = value
        return request_json

    def _collect_existing_event_ids(
        self, client, org_id, session_id, poll_state, seen_event_ids,
    ):
        cursor = poll_state["cursor"]
        while True:
            params = {}
            if cursor is not None:
                params["after"] = cursor
            messages_response = client.get(
                f"{self.BASE_URL}/organizations/{org_id}/sessions/{session_id}/messages",
                params=params,
            )
            messages_response.raise_for_status()
            data = messages_response.json()
            logger.debug(
                "collect existing messages response",
                extra={"data": data},
            )
            for item in data["items"]:
                seen_event_ids.add(item["event_id"])
            has_next_page = data.get("has_next_page")
            new_cursor = data.get("end_cursor")
            if has_next_page:
                if new_cursor is None:
                    raise llm.ModelError(
                        "messages pagination indicated another page"
                        " without an end_cursor"
                    )
                cursor = new_cursor
                poll_state["cursor"] = cursor
                continue
            if new_cursor is not None:
                cursor = new_cursor
            poll_state["cursor"] = cursor
            break

    def _get_session(self, client, org_id, session_id):
        session_response = client.get(
            f"{self.BASE_URL}/organizations/{org_id}/sessions/{session_id}",
        )
        session_response.raise_for_status()
        session_json = session_response.json()
        logger.debug(
            "get_session response",
            extra={"data": session_json},
        )
        return session_json

    def _drain_messages(
        self, client, org_id, session_id, devin_messages, poll_state,
        seen_event_ids,
    ):
        cursor = poll_state["cursor"]
        while True:
            params = {}
            if cursor is not None:
                params["after"] = cursor
            messages_response = client.get(
                f"{self.BASE_URL}/organizations/{org_id}/sessions/{session_id}/messages",
                params=params,
            )
            messages_response.raise_for_status()
            data = messages_response.json()
            logger.debug(
                "messages response",
                extra={"data": data},
            )
            for item in data["items"]:
                if item["event_id"] in seen_event_ids:
                    continue
                seen_event_ids.add(item["event_id"])
                if item["source"] == "devin":
                    devin_message = item["message"]
                    if len(devin_messages) == 0:
                        yield devin_message
                    else:
                        yield "\n" + devin_message
                    devin_messages.append(devin_message)
            has_next_page = data.get("has_next_page")
            new_cursor = data.get("end_cursor")
            if has_next_page:
                if new_cursor is None:
                    raise llm.ModelError(
                        "messages pagination indicated another page"
                        " without an end_cursor"
                    )
                cursor = new_cursor
                poll_state["cursor"] = cursor
                continue

            if new_cursor is not None:
                cursor = new_cursor
                poll_state["cursor"] = cursor
            break
