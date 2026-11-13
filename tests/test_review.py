import json

import httpx2
import pytest
from click.testing import CliRunner
from llm.cli import cli

from conftest import _patched_api

PR_URL = "https://github.com/owner/repo/pull/123"
ENDPOINT = "https://api.devin.ai/v3/organizations/org-test/pr-reviews"
BODY = {
    "status": "pending",
    "repo_path": "github.com/owner/repo",
    "pr_number": 123,
    "commit_sha": "abcd1234",
    "created_at": "2026-01-01T00:00:00Z",
}


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    monkeypatch.setenv("LLM_DEVIN_ORG_ID", "org-test")
    monkeypatch.setenv("LLM_DEVIN_KEY", "test-api-key")


@pytest.fixture
def mock_review_api():
    yield from _patched_api("llm_devin._review")


def invoke(command, *args):
    return CliRunner().invoke(cli, ["devin", "review", command, *args])


@pytest.mark.parametrize("json_output", [False, True])
def test_start(mock_review_api, json_output):
    mock_review_api.post(ENDPOINT, json__eq={"pr_url": PR_URL}).mock(
        return_value=httpx2.Response(200, json=BODY)
    )
    result = invoke("start", PR_URL, *(["--json"] if json_output else []))
    assert result.exit_code == 0, result.output
    assert len(mock_review_api.calls) == 1
    assert mock_review_api.calls[0].method == "POST"
    assert json.loads(mock_review_api.calls[0].content) == {"pr_url": PR_URL}
    if json_output:
        assert json.loads(result.stdout) == {"pr_url": PR_URL, "review": BODY}
    else:
        assert "Devin Review requested." in result.stdout
        for line in [f"PR: {PR_URL}", "Repository: github.com/owner/repo",
                     "PR number: 123", "Commit: abcd1234", "Status: pending",
                     "Created at: 2026-01-01T00:00:00Z", "API returns status only"]:
            assert line in result.stdout


@pytest.mark.parametrize("commit", [None, "ABcd1234"])
@pytest.mark.parametrize("json_output", [False, True])
def test_status(mock_review_api, commit, json_output):
    body = {**BODY, "status": "future-status"}
    mock_review_api.get(ENDPOINT).mock(return_value=httpx2.Response(200, json=body))
    args = [PR_URL]
    if commit:
        args += ["--commit", commit]
    if json_output:
        args += ["--json"]
    result = invoke("status", *args)
    assert result.exit_code == 0, result.output
    assert len(mock_review_api.calls) == 1
    request = mock_review_api.calls[0]
    assert request.method == "GET"
    expected = {"pr_url": PR_URL}
    if commit:
        expected["commit_sha"] = commit
    assert dict(request.url.params) == expected
    if json_output:
        assert json.loads(result.stdout) == {"pr_url": PR_URL, "review": body}
    else:
        assert "Status: future-status" in result.stdout
        assert "requested" not in result.stdout


@pytest.mark.parametrize("commit", [None, "abcd"])
def test_status_not_found(mock_review_api, commit):
    mock_review_api.get(ENDPOINT).mock(return_value=httpx2.Response(404))
    result = invoke("status", PR_URL, *(["--commit", commit] if commit else []))
    assert result.exit_code == 1
    assert result.stdout == ""
    target = f"commit {commit}" if commit else "the current head commit"
    assert f"No Devin Review found for {target} of {PR_URL}" in result.stderr
    assert "llm devin review start" in result.stderr
    assert all(request.method == "GET" for request in mock_review_api.calls)


@pytest.mark.parametrize("command", ["start", "status"])
@pytest.mark.parametrize("code, message", [
    (401, "Check the API key"), (403, "Required permission"),
    (409, "conflict"), (410, "HTTP 410"), (422, "bad field"),
    (429, "rate limited"), (500, "HTTP 500"),
])
def test_api_errors(mock_review_api, command, code, message):
    route = mock_review_api.post if command == "start" else mock_review_api.get
    route(ENDPOINT).mock(return_value=httpx2.Response(
        code, json={"detail": "problem detail", "errors": [{"message": "bad field"}]}
    ))
    result = invoke(command, PR_URL, "--json")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert message in result.stderr
    assert "problem detail" in result.stderr
    if code == 403:
        assert ("UseReviewManual" if command == "start" else "ViewReview") in result.stderr
    assert len(mock_review_api.calls) == 1


def test_start_not_found(mock_review_api):
    mock_review_api.post(ENDPOINT).mock(return_value=httpx2.Response(404))
    result = invoke("start", PR_URL)
    assert result.exit_code == 1
    assert "PR not found or not accessible to Devin" in result.stderr


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo", "ftp://github.com/owner/repo/pull/1",
    "https://gitlab.com/group/project/merge_requests/1", "https://", "not-a-url",
    "https://enterprise.example/", "http://[invalid",
])
@pytest.mark.parametrize("command", ["start", "status"])
def test_invalid_url(mock_review_api, url, command):
    result = invoke(command, url)
    assert result.exit_code == 1
    assert "Invalid PR URL" in result.stderr
    assert mock_review_api.calls == []


@pytest.mark.parametrize("url", [
    "https://enterprise.example/custom/path", "https://gitlab.com/group/project/-/merge_requests/1",
    "https://gitlab.com/group/subgroup/project/-/merge_requests/1/",
])
def test_supported_hosts(mock_review_api, url):
    mock_review_api.post(ENDPOINT, json__eq={"pr_url": url}).mock(
        return_value=httpx2.Response(200, json=BODY)
    )
    result = invoke("start", url)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("commit", ["", "abc", "g123", "a" * 65])
def test_bad_commit(mock_review_api, commit):
    result = invoke("status", PR_URL, "--commit", commit)
    assert result.exit_code == 1
    assert "Invalid commit" in result.stderr
    assert mock_review_api.calls == []


@pytest.mark.parametrize("body", [None, [], {}, {k: v for k, v in BODY.items() if k != "commit_sha"}])
def test_malformed_body(mock_review_api, body):
    mock_review_api.get(ENDPOINT).mock(return_value=httpx2.Response(200, json=body))
    result = invoke("status", PR_URL, "--json")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "unexpected response" in result.stderr


def test_non_json(mock_review_api):
    mock_review_api.post(ENDPOINT).mock(return_value=httpx2.Response(200, text="not json"))
    result = invoke("start", PR_URL)
    assert result.exit_code == 1
    assert "unexpected response" in result.stderr


@pytest.mark.parametrize("command", ["start", "status"])
@pytest.mark.parametrize("error", [httpx2.ConnectError, httpx2.ConnectTimeout])
def test_network_error(mock_review_api, command, error):
    route = mock_review_api.post if command == "start" else mock_review_api.get
    route(ENDPOINT).mock(side_effect=[error("offline")])
    result = invoke(command, PR_URL, "--json")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Could not reach the Devin API" in result.stderr
    assert error.__name__ in result.stderr
    assert "outcome is unknown" not in result.stderr
    assert len(mock_review_api.calls) == 1


@pytest.mark.parametrize("command", ["start", "status"])
@pytest.mark.parametrize("error", [
    httpx2.ReadTimeout, httpx2.WriteTimeout, httpx2.RemoteProtocolError,
])
def test_uncertain_network_outcome(mock_review_api, command, error):
    route = mock_review_api.post if command == "start" else mock_review_api.get
    route(ENDPOINT).mock(side_effect=[error("connection interrupted")])
    result = invoke(command, PR_URL, "--json")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert error.__name__ in result.stderr
    assert len(mock_review_api.calls) == 1
    if command == "start":
        assert "request outcome is unknown" in result.stderr
        assert "review may have started" in result.stderr
        assert f"llm devin review status {PR_URL}" in result.stderr
        assert "before retrying" in result.stderr
        assert "Could not reach" not in result.stderr
    else:
        assert "Could not reach the Devin API" in result.stderr
        assert "outcome is unknown" not in result.stderr


@pytest.mark.parametrize("variable, message", [
    ("LLM_DEVIN_KEY", "No Devin API key found"),
    ("LLM_DEVIN_ORG_ID", "LLM_DEVIN_ORG_ID"),
])
def test_missing_configuration(mock_review_api, monkeypatch, variable, message):
    monkeypatch.delenv(variable)
    result = invoke("start", PR_URL)
    assert result.exit_code == 1
    assert message in result.stderr
    assert mock_review_api.calls == []
