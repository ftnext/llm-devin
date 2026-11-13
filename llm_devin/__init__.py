from __future__ import annotations

import logging

import llm

from llm_devin._cli import register_commands
from llm_devin._deepwiki import DeepWikiClient, DeepWikiModel
from llm_devin._devin import DevinModel

__all__ = [
    "DeepWikiClient",
    "DeepWikiModel",
    "DevinModel",
    "register_commands",
    "register_tools",
]

if hasattr(llm, "Toolbox"):
    from llm_devin._tools import Devin

    __all__.append("Devin")

# Suppress "Unknown SSE event: ping" from DeepWiki MCP server.
# ref: https://github.com/modelcontextprotocol/python-sdk/blob/v1.9.2/src/mcp/client/sse.py#L113-L116
_mcp_sse_logger = logging.getLogger("mcp.client.sse")
_mcp_sse_logger.addHandler(logging.NullHandler())
_mcp_sse_logger.propagate = False


@llm.hookimpl
def register_models(register):
    register(DevinModel())
    register(DeepWikiModel())


@llm.hookimpl
def register_tools(register):
    register(Devin)
