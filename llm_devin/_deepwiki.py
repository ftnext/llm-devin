from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import llm
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from pydantic import Field

if TYPE_CHECKING:
    from mcp.types import CallToolResult

logger = logging.getLogger(__name__)


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
