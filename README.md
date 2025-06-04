# llm-devin

[![PyPI](https://img.shields.io/pypi/v/llm-devin.svg)](https://pypi.org/project/llm-devin/)
[![Changelog](https://img.shields.io/github/v/release/ftnext/llm-devin?include_prereleases&label=changelog)](https://github.com/ftnext/llm-devin/releases)
[![Tests](https://github.com/ftnext/llm-devin/actions/workflows/test.yml/badge.svg)](https://github.com/ftnext/llm-devin/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ftnext/llm-devin/blob/main/LICENSE)



## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-devin
```
## Usage

### Devin API

**prerequisite**: Devin API key (Devin Team Plan)  
https://docs.devin.ai/api-reference/overview#get-an-api-key

```bash
export LLM_DEVIN_KEY=your_api_key_here

llm -m devin "Hello, Devin"
```

### DeepWiki

```bash
llm -m deepwiki -o repository ftnext/llm-devin "Summarize this repository"
```

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
