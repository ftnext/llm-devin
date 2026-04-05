from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import TYPE_CHECKING

import httpx
import llm
# from happy_python_logging.app import configureLogger
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from pydantic import Field

if TYPE_CHECKING:
    from mcp.types import CallToolResult

# Suppress "Unknown SSE event: ping" from DeepWiki MCP server.
# ref: https://github.com/modelcontextprotocol/python-sdk/blob/v1.9.2/src/mcp/client/sse.py#L113-L116
logging.getLogger("mcp.client.sse").addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)
# logger = configureLogger(
#     __name__,
#     level=logging.DEBUG,
#     format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
# )

TIMEOUT = httpx.Timeout(5.0, read=10.0)


def print_immediately(*objects) -> None:
    # ref: https://github.com/simonw/llm/blob/0.26/llm/cli.py#L867-L868
    print(*objects)
    sys.stdout.flush()


class DevinModel(llm.KeyModel):
    needs_key = "devin"
    key_env_var = "LLM_DEVIN_KEY"
    can_stream = True

    BASE_URL = "https://api.devin.ai/v3"

    def __init__(self) -> None:
        self.model_id = "devin"

    def _org_id(self) -> str:
        org_id = os.environ.get("LLM_DEVIN_ORG_ID", "")
        if not org_id:
            raise llm.ModelError(
                "LLM_DEVIN_ORG_ID environment variable is required"
            )
        return org_id

    def execute(self, prompt, stream, response, conversation, key):
        org_id = self._org_id()
        headers = {"Authorization": f"Bearer {key}"}
        request_json = {"prompt": prompt.prompt}
        logger.debug("Request JSON: %s", request_json)
        create_session_response = httpx.post(
            f"{self.BASE_URL}/organizations/{org_id}/sessions",
            headers=headers,
            json=request_json,
            timeout=TIMEOUT,
        )
        create_session_response.raise_for_status()

        session_id = create_session_response.json()["session_id"]
        print_immediately("Devin URL:", create_session_response.json()["url"])

        messages_cursor: str | None = None

        devin_messages: list[str] = []
        while True:
            try:
                session_detail = self._get_session(headers, org_id, session_id)
            except (httpx.RequestError, httpx.HTTPStatusError):
                pass
            else:
                try:
                    messages_response = self._get_messages(
                        headers, org_id, session_id, after=messages_cursor
                    )
                except (httpx.RequestError, httpx.HTTPStatusError):
                    pass
                else:
                    for item in messages_response["items"]:
                        if item["source"] == "devin":
                            devin_message = item["message"]
                            if len(devin_messages) == 0:
                                yield devin_message
                            else:
                                yield "\n" + devin_message
                            devin_messages.append(devin_message)
                    messages_cursor = messages_response.get("end_cursor")

                status = session_detail["status"]
                status_detail = session_detail.get("status_detail")
                if status in {"exit", "error", "suspended"}:
                    break
                if status == "running" and status_detail in {
                    "finished",
                    "waiting_for_user",
                }:
                    break
            time.sleep(5)

    def _get_session(self, headers, org_id, session_id):
        session_response = httpx.get(
            f"{self.BASE_URL}/organizations/{org_id}/sessions/{session_id}",
            headers=headers,
            timeout=TIMEOUT,
        )
        session_response.raise_for_status()
        session_json = session_response.json()
        logger.debug("Session detail: %s", session_json)
        return session_json

    def _get_messages(self, headers, org_id, session_id, *, after=None):
        params = {}
        if after is not None:
            params["after"] = after
        messages_response = httpx.get(
            f"{self.BASE_URL}/organizations/{org_id}/sessions/{session_id}/messages",
            headers=headers,
            params=params,
            timeout=TIMEOUT,
        )
        messages_response.raise_for_status()
        return messages_response.json()


class DeepWikiClient:
    SERVER_URL = "https://mcp.deepwiki.com/sse"

    async def run(self, repository: str, question: str) -> CallToolResult:
        async with sse_client(url=self.SERVER_URL, timeout=60) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.debug("Initialized DeepWiki MCP session")

                arguments = {"repoName": repository, "question": question}
                logger.debug("Calling ask_question tool with arguments: %s", arguments)

                return await session.call_tool("ask_question", arguments)


class DeepWikiModel(llm.Model):
    model_id = "deepwiki"
    needs_key = False
    can_stream = False

    class Options(llm.Options):
        repository: str = Field(
            description="GitHub repository URL: owner/repo",
        )

    def execute(self, prompt, stream, response, conversation):
        client = DeepWikiClient()
        result = asyncio.run(client.run(prompt.options.repository, prompt.prompt))

        for content in result.content:
            if content.type == "text":
                yield content.text


@llm.hookimpl
def register_models(register):
    register(DevinModel())
    register(DeepWikiModel())
