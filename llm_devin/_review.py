from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

import click
import httpx2

from llm_devin._cli import _echo_json, _problem_detail, _resolve_key, _resolve_org_id
from llm_devin._devin import API_BASE_URL, create_http_client


def _validate_pr_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
    except ValueError:
        host = None
    if not host or parsed.scheme not in {"http", "https"}:
        raise click.ClickException("Invalid PR URL: expected an http/https URL with a host.")
    path = parsed.path.rstrip("/")
    valid = bool(path)
    if host == "github.com":
        valid = re.fullmatch(r"/[^/]+/[^/]+/pull/[0-9]+", path) is not None
    elif host == "gitlab.com":
        valid = re.fullmatch(r"/(?:[^/]+/)+[^/]+/-/merge_requests/[0-9]+", path) is not None
    if not valid:
        raise click.ClickException("Invalid PR URL: expected a pull/merge request path.")


def _review_error(response, pr_url, commit, start):
    code = response.status_code
    detail = _problem_detail(response)
    if code == 401:
        message = "Devin API authentication failed (HTTP 401). Check the API key."
    elif code == 403:
        permission = "UseReviewManual" if start else "ViewReview"
        message = f"Devin API access denied (HTTP 403). Required permission: {permission}."
    elif code == 404:
        if start:
            message = f"PR not found or not accessible to Devin: {pr_url}."
        else:
            target = f"commit {commit}" if commit else "the current head commit"
            message = (
                f"No Devin Review found for {target} of {pr_url}."
                " Run 'llm devin review start' to request a review."
            )
    elif code == 409:
        message = "Devin Review conflict (HTTP 409)."
    elif code == 422:
        message = "Devin Review validation failed (HTTP 422)."
        try:
            errors = response.json().get("errors")
        except (ValueError, AttributeError):
            errors = None
        if errors:
            message += f" Errors: {json.dumps(errors, ensure_ascii=False)}"
    elif code == 429:
        message = "Devin API rate limited (HTTP 429)."
    else:
        message = f"Devin API request failed: HTTP {code}."
    if detail:
        message += f" {detail}"
    return click.ClickException(message)


def _request_review(pr_url, commit, json_output, key, start):
    _validate_pr_url(pr_url)
    if commit is not None and re.fullmatch(r"[0-9a-fA-F]{4,64}", commit) is None:
        raise click.ClickException("Invalid commit: expected 4–64 hexadecimal characters.")
    org_id = _resolve_org_id()
    headers = {"Authorization": f"Bearer {_resolve_key(key)}"}
    endpoint = f"{API_BASE_URL}/organizations/{org_id}/pr-reviews"
    try:
        with create_http_client(headers) as client:
            if start:
                response = client.post(endpoint, json={"pr_url": pr_url})
            else:
                params = {"pr_url": pr_url}
                if commit is not None:
                    params["commit_sha"] = commit
                response = client.get(endpoint, params=params)
            response.raise_for_status()
    except httpx2.HTTPStatusError as ex:
        raise _review_error(ex.response, pr_url, commit, start) from ex
    except httpx2.RequestError as ex:
        if start and not isinstance(ex, (httpx2.ConnectError, httpx2.ConnectTimeout)):
            raise click.ClickException(
                f"Devin Review request outcome is unknown ({type(ex).__name__})."
                " The review may have started."
                f" Run 'llm devin review status {pr_url}' before retrying."
            ) from ex
        raise click.ClickException(
            f"Could not reach the Devin API ({type(ex).__name__})"
        ) from ex
    try:
        body = response.json()
    except ValueError as ex:
        raise click.ClickException("The Devin API returned an unexpected response.") from ex
    fields = ("status", "repo_path", "pr_number", "commit_sha", "created_at")
    if not isinstance(body, dict) or any(field not in body for field in fields):
        raise click.ClickException("The Devin API returned an unexpected response.")
    if json_output:
        _echo_json({"pr_url": pr_url, "review": body})
        return
    if start:
        click.echo("Devin Review requested.")
    click.echo(f"PR: {pr_url}")
    for label, field in (
        ("Repository", "repo_path"),
        ("PR number", "pr_number"),
        ("Commit", "commit_sha"),
        ("Status", "status"),
        ("Created at", "created_at"),
    ):
        click.echo(f"{label}: {body[field]}")
    click.echo("Findings are posted on the PR itself; the API returns status only.")


@click.group(help="Trigger or check Devin Review for a pull/merge request")
def review():
    pass


@review.command()
@click.argument("pr_url")
@click.option("json_output", "--json", is_flag=True, help="Output JSON")
@click.option("--key", help="Devin API key to use")
def start(pr_url, json_output, key):
    """Request a review of the latest PR head commit (no retries)."""
    _request_review(pr_url, None, json_output, key, start=True)


@review.command()
@click.argument("pr_url")
@click.option("--commit", help="Commit SHA or hexadecimal prefix (4–64 characters)")
@click.option("json_output", "--json", is_flag=True, help="Output JSON")
@click.option("--key", help="Devin API key to use")
def status(pr_url, commit, json_output, key):
    """Read review status for a commit (default: current PR head)."""
    _request_review(pr_url, commit, json_output, key, start=False)
