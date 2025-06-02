import logging
import time

import httpx
import llm
from happy_python_logging.app import configureLogger

logger = configureLogger(
    __name__,
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
)

TIMEOUT = httpx.Timeout(5.0, read=10.0)


class DevinModel(llm.KeyModel):
    needs_key = "devin"
    key_env_var = "LLM_DEVIN_KEY"
    can_stream = False

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
        print("Devin URL:", create_session_response.json()["url"])

        while True:
            session_detail = httpx.get(
                f"https://api.devin.ai/v1/session/{session_id}",
                headers=headers,
                timeout=TIMEOUT,
            )
            session_detail.raise_for_status()
            session_detail_json = session_detail.json()
            logger.debug("Session detail: %s", session_detail_json)
            if session_detail_json["status_enum"] in {"blocked", "stopped", "finished"}:
                break
            time.sleep(5)

        for message in session_detail_json["messages"]:
            if message["type"] == "devin_message":
                yield message["message"]


class DeepWikiMCP(llm.Toolbox):
    """MCP integration for DeepWiki repository analysis"""
    
    def __init__(self, repository: str):
        self.repository = repository
        self.server_url = "https://mcp.deepwiki.com/sse"
    
    async def ask_question(self, question: str) -> str:
        """Ask a question about the configured repository using DeepWiki MCP"""
        from mcp.client.session import ClientSession
        from mcp.client.sse import sse_client
        
        try:
            async with sse_client(url=self.server_url, timeout=60) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    arguments = {"repoName": self.repository, "question": question}
                    result = await session.call_tool("ask_question", arguments)
                    
                    if hasattr(result, "content"):
                        response_parts = []
                        for content in result.content:
                            if content.type == "text":
                                response_parts.append(content.text)
                            else:
                                response_parts.append(str(content))
                        return "\n".join(response_parts)
                    else:
                        return str(result)
                        
        except Exception as e:
            return f"Error connecting to DeepWiki MCP server: {e}"


@llm.hookimpl
def register_models(register):
    register(DevinModel())


@llm.hookimpl
def register_tools(register):
    register(DeepWikiMCP)
