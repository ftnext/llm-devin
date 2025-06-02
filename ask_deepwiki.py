# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mcp",
# ]
# ///

import argparse
import asyncio
import traceback

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


class DeepWikiClient:
    SERVER_URL = "https://mcp.deepwiki.com/sse"

    def __init__(self, repository: str) -> None:
        self.repository = repository
        self.session: ClientSession | None = None

    async def connect(self):
        print(f"🔗 Connecting to DeepWiki MCP server at {self.SERVER_URL}...")

        try:
            async with sse_client(
                url=self.SERVER_URL,
                timeout=60,
            ) as (read_stream, write_stream):
                await self._run_session(read_stream, write_stream)

        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            traceback.print_exc()

    async def _run_session(self, read_stream, write_stream):
        print("🤝 Initializing MCP session...")
        async with ClientSession(read_stream, write_stream) as session:
            self.session = session
            print("⚡ Starting session initialization...")
            await session.initialize()
            print("✨ Session initialization complete!")
            print("✅ Connected to DeepWiki MCP server")

            question = input("Ask: ")

            await self.ask_question(repo_name=self.repository, question=question)

    async def ask_question(self, repo_name: str, question: str):
        if not self.session:
            print("❌ Not connected to server")
            return

        try:
            arguments = {"repoName": repo_name, "question": question}

            print("\n🔧 Calling ask_question tool...")
            print(f"   Repository: {repo_name}")
            print(f"   Question: {question}")

            result = await self.session.call_tool("ask_question", arguments)

            print("\n📝 Answer:")
            if hasattr(result, "content"):
                for content in result.content:
                    if content.type == "text":
                        print(content.text)
                    else:
                        print(content)
            else:
                print(result)

        except Exception as e:
            print(f"❌ Failed to call ask_question tool: {e}")


async def main(repository: str):
    print("🚀 DeepWiki MCP Client")
    print("Connecting to DeepWiki MCP server...")

    client = DeepWikiClient(repository)
    await client.connect()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("repository")
    args = parser.parse_args()

    asyncio.run(main(args.repository))


if __name__ == "__main__":
    cli()
