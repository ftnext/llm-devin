from __future__ import annotations

import json

import click
import httpx2
import llm

from llm_devin._devin import (
    API_BASE_URL,
    create_http_client,
    parse_session_reference,
    resolve_org_id,
    session_url,
)

MESSAGES_PAGE_SIZE = 200


def _resolve_session_id(reference: str) -> str:
    try:
        return parse_session_reference(reference)
    except llm.ModelError as ex:
        raise click.ClickException(str(ex)) from ex


def _resolve_org_id() -> str:
    try:
        return resolve_org_id()
    except llm.ModelError as ex:
        raise click.ClickException(str(ex)) from ex


def _resolve_key(key: str | None) -> str:
    resolved = llm.get_key(key, "devin", "LLM_DEVIN_KEY")
    if not resolved:
        raise click.ClickException(
            "No Devin API key found."
            " Set the LLM_DEVIN_KEY environment variable,"
            " store one with 'llm keys set devin', or pass --key."
        )
    return resolved


def _problem_detail(response: httpx2.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        return body["detail"]
    return None


def _read_error(session_id: str, ex: Exception) -> click.ClickException:
    url = session_url(session_id)
    if isinstance(ex, httpx2.HTTPStatusError):
        status_code = ex.response.status_code
        detail = _problem_detail(ex.response)
        suffix = f" ({detail})" if detail else ""
        if status_code == 401:
            return click.ClickException(
                f"Devin API authentication failed: HTTP 401{suffix}."
                " Check the API key."
            )
        if status_code in {403, 404, 410}:
            return click.ClickException(
                f"Could not read Devin session {session_id} ({url}):"
                f" HTTP {status_code}{suffix}."
                " The session may not exist, may have been deleted,"
                " or may not be accessible with this API key."
            )
        return click.ClickException(
            f"Devin API request failed for {session_id} ({url}):"
            f" HTTP {status_code}{suffix}."
        )
    return click.ClickException(
        f"Could not reach the Devin API for {session_id} ({url}):"
        f" {type(ex).__name__}: {ex}."
    )


def _invalid_response_error(session_id: str) -> click.ClickException:
    return click.ClickException(
        f"The Devin API returned an unexpected response for {session_id}"
        f" ({session_url(session_id)})."
    )


def _json_body(response: httpx2.Response, session_id: str) -> dict:
    try:
        body = response.json()
    except ValueError as ex:
        raise _invalid_response_error(session_id) from ex
    if not isinstance(body, dict):
        raise _invalid_response_error(session_id)
    return body


def _require_fields(body: dict, fields: tuple[str, ...], session_id: str) -> dict:
    if any(field not in body for field in fields):
        raise _invalid_response_error(session_id)
    return body


def _get_session(client, org_id: str, session_id: str) -> dict:
    try:
        response = client.get(
            f"{API_BASE_URL}/organizations/{org_id}/sessions/{session_id}",
        )
        response.raise_for_status()
    except (httpx2.RequestError, httpx2.HTTPStatusError) as ex:
        raise _read_error(session_id, ex) from ex
    return _require_fields(
        _json_body(response, session_id),
        (
            "session_id",
            "url",
            "status",
            "created_at",
            "updated_at",
            "acus_consumed",
        ),
        session_id,
    )


def _get_messages(client, org_id: str, session_id: str) -> list[dict]:
    items: list[dict] = []
    cursor = None
    while True:
        params: dict = {"first": MESSAGES_PAGE_SIZE}
        if cursor is not None:
            params["after"] = cursor
        try:
            response = client.get(
                f"{API_BASE_URL}/organizations/{org_id}/sessions/{session_id}/messages",
                params=params,
            )
            response.raise_for_status()
        except (httpx2.RequestError, httpx2.HTTPStatusError) as ex:
            raise _read_error(session_id, ex) from ex
        data = _json_body(response, session_id)
        if not isinstance(data.get("items"), list):
            raise _invalid_response_error(session_id)
        for item in data["items"]:
            if not isinstance(item, dict):
                raise _invalid_response_error(session_id)
            _require_fields(item, ("source", "message"), session_id)
        items.extend(data["items"])
        has_next_page = data.get("has_next_page")
        if not isinstance(has_next_page, bool):
            raise _invalid_response_error(session_id)
        if not has_next_page:
            break
        cursor = data.get("end_cursor")
        if not isinstance(cursor, str) or not cursor:
            raise click.ClickException(
                "The Devin API reported another page of messages"
                " without an end_cursor."
            )
    return items


def _latest_devin_message(messages: list[dict]) -> dict | None:
    for message in reversed(messages):
        if message["source"] == "devin":
            return message
    return None


def _echo_json(payload) -> None:
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _echo_status(session: dict, latest: dict | None) -> None:
    click.echo(f"Session: {session['session_id']}")
    click.echo(f"URL: {session['url']}")
    status = session["status"]
    status_detail = session.get("status_detail")
    if status_detail:
        click.echo(f"Status: {status} ({status_detail})")
    else:
        click.echo(f"Status: {status}")
    title = session.get("title")
    if title:
        click.echo(f"Title: {title}")
    click.echo(f"Created at: {session['created_at']}")
    click.echo(f"Updated at: {session['updated_at']}")
    click.echo(f"ACUs consumed: {session['acus_consumed']}")
    tags = session.get("tags")
    if tags:
        click.echo(f"Tags: {', '.join(tags)}")
    structured_output = session.get("structured_output")
    if structured_output is not None:
        click.echo("Structured output:")
        click.echo(
            json.dumps(structured_output, indent=2, ensure_ascii=False)
        )
    pull_requests = session.get("pull_requests") or []
    if pull_requests:
        click.echo("Pull requests:")
        for pull_request in pull_requests:
            state = pull_request.get("pr_state")
            state_suffix = f" ({state})" if state else ""
            click.echo(f"  {pull_request.get('pr_url')}{state_suffix}")
    else:
        click.echo("Pull requests: none")
    if latest is None:
        click.echo("Latest Devin message: none")
    else:
        click.echo("Latest Devin message:")
        click.echo(latest["message"])


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def devin():
        "Read the state of an existing Devin session"

    @devin.command()
    @click.argument("session")
    @click.option("json_output", "--json", is_flag=True, help="Output JSON")
    @click.option("--key", help="Devin API key to use")
    def status(session, json_output, key):
        """
        Show the status of an existing Devin session

        SESSION is a Devin session ID or its app.devin.ai URL.
        """
        session_id = _resolve_session_id(session)
        org_id = _resolve_org_id()
        headers = {"Authorization": f"Bearer {_resolve_key(key)}"}
        with create_http_client(headers) as client:
            session_detail = _get_session(client, org_id, session_id)
            messages = _get_messages(client, org_id, session_id)
        latest = _latest_devin_message(messages)
        if json_output:
            _echo_json({"session": session_detail, "latest_devin_message": latest})
        else:
            _echo_status(session_detail, latest)

    @devin.command()
    @click.argument("session")
    @click.option("json_output", "--json", is_flag=True, help="Output JSON")
    @click.option(
        "-n",
        "--limit",
        type=click.IntRange(min=1),
        help="Show only the most recent N messages",
    )
    @click.option(
        "--source",
        type=click.Choice(["all", "devin", "user"]),
        default="all",
        show_default=True,
        help="Filter messages by source",
    )
    @click.option("--key", help="Devin API key to use")
    def messages(session, json_output, limit, source, key):
        """
        Show the messages of an existing Devin session

        SESSION is a Devin session ID or its app.devin.ai URL.
        Messages are listed oldest first.
        """
        session_id = _resolve_session_id(session)
        org_id = _resolve_org_id()
        headers = {"Authorization": f"Bearer {_resolve_key(key)}"}
        with create_http_client(headers) as client:
            items = _get_messages(client, org_id, session_id)
        if source != "all":
            items = [item for item in items if item["source"] == source]
        if limit is not None:
            items = items[-limit:]
        if json_output:
            _echo_json({"session_id": session_id, "messages": items})
            return
        if not items:
            click.echo("No messages")
            return
        for item in items:
            click.echo(f"[{item['source']}] {item['message']}")
