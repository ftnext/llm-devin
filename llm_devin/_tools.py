from __future__ import annotations

import re
import threading

import httpx2
import llm

from llm_devin import _devin

DEFAULT_MAX_ACU_LIMIT = 5
MAX_ACU_LIMIT_CEILING = 50
MAX_SESSIONS_CEILING = 10
DELEGATION_DEVIN_MODES = (
    "normal",
    "lite",
    "fusion",
    "swe-2-medium",
    "swe-2-high",
    "swe-2-max",
)
MESSAGES_PAGE_SIZE = 200
MAX_MESSAGE_PAGES = 50
MAX_MESSAGES_LIMIT = 20
MAX_MESSAGE_CHARS = 4000
MAX_DETAIL_CHARS = 200
MAX_PROMPT_CHARS = 30000
MAX_TITLE_CHARS = 200
REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SESSION_REQUIRED_FIELDS = (
    "session_id",
    "url",
    "status",
    "created_at",
    "updated_at",
    "acus_consumed",
)
STILL_WORKING_HINT = (
    "Devin is still working asynchronously. Do not poll in a loop:"
    " share the session URL with the user and check again later."
)


class DevinToolError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        reason: str | None = None,
        request_sent: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.request_sent = request_sent


def _bounded_int(value, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DevinToolError(f"{name} must be an integer")
    if not low <= value <= high:
        raise DevinToolError(f"{name} must be between {low} and {high}")
    return value


def _validate_mode(value: str, name: str = "devin_mode") -> str:
    if value not in DELEGATION_DEVIN_MODES:
        raise DevinToolError(
            f"{name} must be one of: {', '.join(DELEGATION_DEVIN_MODES)}"
        )
    return value


def _parse_repos(value: str) -> list[str]:
    repos = [item.strip() for item in value.split(",") if item.strip()]
    for repo in repos:
        if not REPO_PATTERN.match(repo):
            raise DevinToolError(
                f"Invalid repository {repo!r} (expected owner/repo)"
            )
    return repos


def _session_id(reference: str) -> str:
    if not isinstance(reference, str):
        raise DevinToolError("session must be a Devin session ID or URL")
    try:
        return _devin.parse_session_reference(reference)
    except llm.ModelError as ex:
        raise DevinToolError(str(ex)) from None


def _api_key() -> str:
    key = llm.get_key(None, "devin", "LLM_DEVIN_KEY")
    if not key:
        raise DevinToolError(
            "No Devin API key found. Set the LLM_DEVIN_KEY environment"
            " variable or store one with 'llm keys set devin'."
        )
    return key


def _org_id() -> str:
    try:
        return _devin.resolve_org_id()
    except llm.ModelError as ex:
        raise DevinToolError(str(ex)) from None


def _problem_detail(response: httpx2.Response, secret: str) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        return body["detail"].replace(secret, "[redacted]")[:MAX_DETAIL_CHARS]
    return None


def _request_error(action: str, ex: Exception, secret: str) -> DevinToolError:
    if isinstance(ex, httpx2.HTTPStatusError):
        status_code = ex.response.status_code
        detail = _problem_detail(ex.response, secret)
        suffix = f" ({detail})" if detail else ""
        if status_code == 401:
            hint = " Check the Devin API key."
        elif status_code in {403, 404, 410} and action.startswith("read"):
            hint = (
                " The session may not exist or may not be accessible"
                " with this API key."
            )
        else:
            hint = ""
        return DevinToolError(
            f"Could not {action}: Devin API returned HTTP {status_code}"
            f"{suffix}.{hint}",
            status_code=status_code,
            reason=f"HTTP {status_code}{suffix}",
            request_sent=True,
        )
    reason = type(ex).__name__
    if isinstance(ex, (httpx2.ConnectError, httpx2.ConnectTimeout)):
        return DevinToolError(
            f"Could not {action}: could not reach the Devin API ({reason}).",
            reason=reason,
        )
    return DevinToolError(
        f"Could not {action}: no complete response from the Devin API"
        f" ({reason}).",
        reason=reason,
        request_sent=True,
    )


def _uncertain_start_error(reason: str) -> DevinToolError:
    return DevinToolError(
        f"Could not confirm whether a Devin session was started ({reason})."
        " A session may already exist and consume ACUs: check the Devin web"
        " app before retrying. This attempt counts toward the session limit."
    )


def _unexpected_response(action: str) -> DevinToolError:
    return DevinToolError(
        f"Could not {action}: unexpected response from the Devin API.",
        reason="unexpected response",
        request_sent=True,
    )


def _json_object(response: httpx2.Response, action: str) -> dict:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise _unexpected_response(action)
    return body


class _Api:
    def __init__(self):
        key = _api_key()
        self._key = key
        self._base_url = f"{_devin.API_BASE_URL}/organizations/{_org_id()}"
        self._client = _devin.create_http_client(
            {"Authorization": f"Bearer {key}"}
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._client.close()

    def request(self, method: str, path: str, action: str, **kwargs) -> dict:
        try:
            response = self._client.request(
                method, self._base_url + path, **kwargs
            )
            response.raise_for_status()
        except (httpx2.RequestError, httpx2.HTTPStatusError) as ex:
            raise _request_error(action, ex, self._key) from None
        return _json_object(response, action)


def _is_finished(status: str | None, status_detail: str | None) -> bool:
    if status in {"exit", "error", "suspended"}:
        return True
    return status == "running" and status_detail in {
        "finished",
        "waiting_for_user",
        "waiting_for_approval",
    }


class Devin(llm.Toolbox):
    """
    Delegate tasks to cloud Devin sessions (Devin API v3).

    Starting a session costs Devin ACUs and is capped by max_acu_limit.
    Status and messages are read-only.
    """

    def __init__(
        self,
        max_acu_limit: int = DEFAULT_MAX_ACU_LIMIT,
        devin_mode: str | None = None,
        max_sessions: int = 1,
    ):
        try:
            self._max_acu_limit = _bounded_int(
                max_acu_limit, "max_acu_limit", 1, MAX_ACU_LIMIT_CEILING
            )
            self._max_sessions = _bounded_int(
                max_sessions, "max_sessions", 1, MAX_SESSIONS_CEILING
            )
            self._devin_mode = (
                None if devin_mode is None else _validate_mode(devin_mode)
            )
        except DevinToolError as ex:
            raise ValueError(f"Devin toolbox: {ex}") from None
        self._started: list[str] = []
        self._attempts = 0
        self._budget_lock = threading.Lock()

    def start_session(
        self,
        prompt: str,
        devin_mode: str | None = None,
        repos: str | None = None,
        title: str | None = None,
        max_acu_limit: int | None = None,
    ) -> dict:
        """
        Delegate a task to a new cloud Devin session and return immediately.

        Devin is an autonomous software engineering agent, not a raw model:
        it works asynchronously for minutes or hours and consumes ACUs.
        This returns the session ID and URL without waiting. Check progress
        later with Devin_session_status and read replies with
        Devin_session_messages instead of starting another session.

        prompt: complete, self-contained task description.
        devin_mode: optional agent mode (normal, lite, fusion, swe-2-medium,
        swe-2-high, swe-2-max). Omit to use the configured or organization
        default.
        repos: optional comma-separated owner/repo list.
        title: optional session title.
        max_acu_limit: optional ACU cap, at most the configured cap.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise DevinToolError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise DevinToolError(
                f"prompt must be at most {MAX_PROMPT_CHARS} characters"
            )
        limit = self._max_acu_limit
        if max_acu_limit is not None:
            limit = _bounded_int(
                max_acu_limit, "max_acu_limit", 1, self._max_acu_limit
            )
        mode = self._devin_mode
        if devin_mode is not None:
            _validate_mode(devin_mode)
            if mode is not None and devin_mode != mode:
                raise DevinToolError(
                    f"devin_mode is fixed to {mode!r} by the toolbox"
                    " configuration"
                )
            mode = devin_mode
        request_json: dict = {
            "prompt": prompt.strip(),
            "max_acu_limit": limit,
        }
        if mode is not None:
            request_json["devin_mode"] = mode
        if repos:
            parsed_repos = _parse_repos(repos)
            if parsed_repos:
                request_json["repos"] = parsed_repos
        if title and title.strip():
            request_json["title"] = title.strip()[:MAX_TITLE_CHARS]

        with _Api() as api:
            self._reserve_session()
            try:
                body = api.request(
                    "POST",
                    "/sessions",
                    "start a Devin session",
                    json=request_json,
                )
            except DevinToolError as ex:
                if ex.status_code is not None and 400 <= ex.status_code < 500:
                    with self._budget_lock:
                        self._attempts -= 1
                    raise
                if ex.request_sent:
                    raise _uncertain_start_error(ex.reason) from None
                raise
        session_id = body.get("session_id")
        url = body.get("url")
        if not _is_session_reference(session_id, url):
            raise _uncertain_start_error("unexpected response")
        with self._budget_lock:
            self._started.append(session_id)
        return {
            "session_id": session_id,
            "url": url,
            "status": body.get("status"),
            "devin_mode": body.get("devin_mode", mode),
            "max_acu_limit": limit,
            "note": (
                "Devin started working asynchronously. Share the URL with"
                " the user. Check later with Devin_session_status."
            ),
        }

    def _reserve_session(self) -> None:
        with self._budget_lock:
            if self._attempts >= self._max_sessions:
                started = ", ".join(self._started) or "none confirmed"
                raise DevinToolError(
                    f"This Devin toolbox may start at most {self._max_sessions}"
                    f" session(s) (started: {started})."
                    " Use Devin_session_status or Devin_session_messages"
                    " instead."
                )
            self._attempts += 1

    def session_status(self, session: str) -> dict:
        """
        Read the current status of a Devin session (read-only, one request).

        session: Devin session ID (devin-...) or app.devin.ai session URL.
        Returns status, status_detail, ACUs consumed so far, pull requests
        and structured output. Does not wait for the session to finish.
        """
        session_id = _session_id(session)
        action = "read the Devin session"
        with _Api() as api:
            body = api.request("GET", f"/sessions/{session_id}", action)
        if not isinstance(body.get("status"), str) or any(
            field not in body for field in SESSION_REQUIRED_FIELDS
        ):
            raise _unexpected_response(action)
        status = body.get("status")
        status_detail = body.get("status_detail")
        result = {
            "session_id": body.get("session_id", session_id),
            "url": body.get("url", _devin.session_url(session_id)),
            "status": status,
            "status_detail": status_detail,
            "title": body.get("title"),
            "devin_mode": body.get("devin_mode"),
            "acus_consumed": body.get("acus_consumed"),
            "updated_at": body.get("updated_at"),
            "pull_requests": [
                {"url": pr.get("pr_url"), "state": pr.get("pr_state")}
                for pr in body.get("pull_requests") or []
                if isinstance(pr, dict)
            ],
            "structured_output": body.get("structured_output"),
        }
        if not _is_finished(status, status_detail):
            result["note"] = STILL_WORKING_HINT
        return result

    def session_messages(
        self,
        session: str,
        limit: int = 5,
        source: str = "devin",
    ) -> dict:
        """
        Read the most recent messages of a Devin session (read-only).

        session: Devin session ID (devin-...) or app.devin.ai session URL.
        limit: number of most recent messages to return (1-20).
        source: "devin" (default), "user", or "all".
        """
        session_id = _session_id(session)
        limit = _bounded_int(limit, "limit", 1, MAX_MESSAGES_LIMIT)
        if source not in {"devin", "user", "all"}:
            raise DevinToolError("source must be one of: devin, user, all")
        with _Api() as api:
            items = _list_messages(api, session_id)
        if source != "all":
            items = [item for item in items if item.get("source") == source]
        messages = []
        for item in items[-limit:]:
            text = item["message"]
            entry = {
                "source": item.get("source"),
                "created_at": item.get("created_at"),
                "message": text[:MAX_MESSAGE_CHARS],
            }
            if len(text) > MAX_MESSAGE_CHARS:
                entry["truncated"] = True
            messages.append(entry)
        return {
            "session_id": session_id,
            "url": _devin.session_url(session_id),
            "messages": messages,
        }


def _list_messages(api: _Api, session_id: str) -> list[dict]:
    action = "read the Devin session messages"
    items: list[dict] = []
    cursor = None
    seen_cursors: set[str] = set()
    for _ in range(MAX_MESSAGE_PAGES):
        params: dict = {"first": MESSAGES_PAGE_SIZE}
        if cursor is not None:
            params["after"] = cursor
        data = api.request(
            "GET", f"/sessions/{session_id}/messages", action, params=params
        )
        page = data.get("items")
        has_next_page = data.get("has_next_page")
        if (
            not isinstance(page, list)
            or not isinstance(has_next_page, bool)
            or not all(_is_message(item) for item in page)
        ):
            raise _unexpected_response(action)
        items.extend(page)
        if not has_next_page:
            return items
        next_cursor = data.get("end_cursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            raise DevinToolError(
                f"Could not {action}: the Devin API reported another page"
                " without an end_cursor."
            )
        if next_cursor in seen_cursors:
            raise DevinToolError(
                f"Could not {action}: the Devin API returned the same"
                " page cursor twice."
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise DevinToolError(
        f"Could not {action}: more than"
        f" {MAX_MESSAGE_PAGES * MESSAGES_PAGE_SIZE} messages."
    )


def _is_message(item) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("source"), str)
        and isinstance(item.get("message"), str)
    )


def _is_session_reference(session_id, url) -> bool:
    if not isinstance(session_id, str) or not isinstance(url, str):
        return False
    try:
        return (
            _devin.parse_session_reference(session_id)
            == _devin.parse_session_reference(url)
            == session_id
        )
    except llm.ModelError:
        return False
