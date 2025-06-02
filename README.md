# llm-devin

[![PyPI](https://img.shields.io/pypi/v/llm-devin.svg)](https://pypi.org/project/llm-devin/)
[![Changelog](https://img.shields.io/github/v/release/ftnext/llm-devin?include_prereleases&label=changelog)](https://github.com/ftnext/llm-devin/releases)
[![Tests](https://github.com/ftnext/llm-devin/actions/workflows/test.yml/badge.svg)](https://github.com/ftnext/llm-devin/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ftnext/llm-devin/blob/main/LICENSE)

LLM plugin for Devin AI integration with MCP (Model Context Protocol) support

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-devin
```

## Features

### Devin Model
Chat with Devin AI using the LLM interface.

### DeepWiki MCP Integration
Ask questions about GitHub repositories using the DeepWiki MCP server.

## Usage

### Devin Model

**prerequisite**: Devin API key (Devin Team Plan)  
https://docs.devin.ai/api-reference/overview#get-an-api-key

```bash
export LLM_DEVIN_KEY=your_api_key_here

llm -m devin "Hello, Devin"
```

### DeepWiki MCP Tool

Use the DeepWiki MCP tool to ask questions about repositories:

```bash
llm -T DeepWikiMCP:owner/repo "What is this repository about?"
```

Examples:

```bash
# Ask about a specific repository
llm -T DeepWikiMCP:simonw/llm "What does this library do?"

# Get information about the codebase structure
llm -T DeepWikiMCP:ftnext/llm-devin "How is this project organized?"

# Ask about specific functionality
llm -T DeepWikiMCP:python/cpython "How does the garbage collector work?"
```

The DeepWikiMCP tool connects to the DeepWiki MCP server to provide intelligent answers about repository contents, structure, and functionality.

## Development

To set up this plugin locally, first checkout the code:
```bash
cd llm-devin
```
Then create a new virtual environment and install the dependencies and test dependencies:
```bash
uv sync --extra test
```
To run the tests:
```bash
uv run pytest
```
