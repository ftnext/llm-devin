from __future__ import annotations

import asyncio
import logging
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

    def __init__(self) -> None:
        self.model_id = "devin"

    def execute(self, prompt, stream, response, conversation, key):
        headers = {"Authorization": f"Bearer {key}"}
        request_json = {"prompt": prompt.prompt, "idempotent": True}
        logger.debug("Request JSON: %s", request_json)
        create_session_response = httpx.post(
            "https://api.devin.ai/v1/sessions",
            headers=headers,
            json=request_json,
            timeout=TIMEOUT,
        )
        create_session_response.raise_for_status()

        session_id = create_session_response.json().get("session_id")
        print_immediately("Devin URL:", create_session_response.json()["url"])

        yielded_message_count = 0

        devin_messages = []
        while True:
            try:
                session_detail = self._get_session_detail(headers, session_id)
            except (httpx.RequestError, httpx.HTTPStatusError):
                pass
            else:
                # Yield new messages incrementally
                messages = session_detail["messages"]
                for i in range(yielded_message_count, len(messages)):
                    message = messages[i]
                    if message["type"] == "devin_message":
                        devin_message = message["message"]
                        if len(devin_messages) == 0:
                            yield devin_message
                        else:
                            yield "\n" + devin_message
                        devin_messages.append(devin_message)
                yielded_message_count = len(messages)

                if session_detail["status_enum"] in {"blocked", "stopped", "finished"}:
                    break
            time.sleep(5)

    def _get_session_detail(self, headers, session_id):
        timeout = httpx.Timeout(5.0, read=10.0)
        session_detail = httpx.get(
            f"https://api.devin.ai/v1/session/{session_id}",
            headers=headers,
            timeout=timeout,
        )
        session_detail.raise_for_status()
        session_detail_json = session_detail.json()
        logger.debug("Session detail: %s", session_detail_json)
        return session_detail_json


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
