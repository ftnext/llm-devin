import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from llm_devin import DeepWikiMCP


@pytest.mark.asyncio
async def test_deepwiki_mcp_ask_question_success():
    """Test successful question asking with DeepWiki MCP"""
    mcp_tool = DeepWikiMCP("owner/repo")
    
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "This repository is a Python library for testing."
    
    mock_result = MagicMock()
    mock_result.content = [mock_content]
    
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    
    with patch("mcp.client.sse.sse_client") as mock_sse_client, \
         patch("mcp.client.session.ClientSession") as mock_client_session:
        
        mock_sse_client.return_value.__aenter__.return_value = ("read_stream", "write_stream")
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        result = await mcp_tool.ask_question("What is this repository about?")
        
        assert result == "This repository is a Python library for testing."
        mock_session.initialize.assert_called_once()
        mock_session.call_tool.assert_called_once_with(
            "ask_question", 
            {"repoName": "owner/repo", "question": "What is this repository about?"}
        )


@pytest.mark.asyncio
async def test_deepwiki_mcp_ask_question_multiple_content():
    """Test question asking with multiple content parts"""
    mcp_tool = DeepWikiMCP("owner/repo")
    
    mock_content1 = MagicMock()
    mock_content1.type = "text"
    mock_content1.text = "Part 1"
    
    mock_content2 = MagicMock()
    mock_content2.type = "text"
    mock_content2.text = "Part 2"
    
    mock_result = MagicMock()
    mock_result.content = [mock_content1, mock_content2]
    
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    
    with patch("mcp.client.sse.sse_client") as mock_sse_client, \
         patch("mcp.client.session.ClientSession") as mock_client_session:
        
        mock_sse_client.return_value.__aenter__.return_value = ("read_stream", "write_stream")
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        result = await mcp_tool.ask_question("Tell me more")
        
        assert result == "Part 1\nPart 2"


@pytest.mark.asyncio
async def test_deepwiki_mcp_ask_question_no_content_attribute():
    """Test question asking when result has no content attribute"""
    mcp_tool = DeepWikiMCP("owner/repo")
    
    mock_result = "Simple string result"
    
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    
    with patch("mcp.client.sse.sse_client") as mock_sse_client, \
         patch("mcp.client.session.ClientSession") as mock_client_session:
        
        mock_sse_client.return_value.__aenter__.return_value = ("read_stream", "write_stream")
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        result = await mcp_tool.ask_question("Simple question")
        
        assert result == "Simple string result"


@pytest.mark.asyncio
async def test_deepwiki_mcp_connection_error():
    """Test error handling when MCP connection fails"""
    mcp_tool = DeepWikiMCP("owner/repo")
    
    with patch("mcp.client.sse.sse_client") as mock_sse_client:
        mock_sse_client.side_effect = Exception("Connection failed")
        
        result = await mcp_tool.ask_question("Any question")
        
        assert result == "Error connecting to DeepWiki MCP server: Connection failed"


@pytest.mark.asyncio
async def test_deepwiki_mcp_session_error():
    """Test error handling when session initialization fails"""
    mcp_tool = DeepWikiMCP("owner/repo")
    
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock(side_effect=Exception("Session init failed"))
    
    with patch("mcp.client.sse.sse_client") as mock_sse_client, \
         patch("mcp.client.session.ClientSession") as mock_client_session:
        
        mock_sse_client.return_value.__aenter__.return_value = ("read_stream", "write_stream")
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        result = await mcp_tool.ask_question("Any question")
        
        assert result == "Error connecting to DeepWiki MCP server: Session init failed"


def test_deepwiki_mcp_initialization():
    """Test DeepWikiMCP initialization"""
    mcp_tool = DeepWikiMCP("test/repo")
    
    assert mcp_tool.repository == "test/repo"
    assert mcp_tool.server_url == "https://mcp.deepwiki.com/sse"


@pytest.mark.asyncio
async def test_deepwiki_mcp_non_text_content():
    """Test handling of non-text content types"""
    mcp_tool = DeepWikiMCP("owner/repo")
    
    mock_content1 = MagicMock()
    mock_content1.type = "text"
    mock_content1.text = "Text content"
    
    mock_content2 = MagicMock()
    mock_content2.type = "image"
    mock_content2.__str__ = MagicMock(return_value="Image content")
    
    mock_result = MagicMock()
    mock_result.content = [mock_content1, mock_content2]
    
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    
    with patch("mcp.client.sse.sse_client") as mock_sse_client, \
         patch("mcp.client.session.ClientSession") as mock_client_session:
        
        mock_sse_client.return_value.__aenter__.return_value = ("read_stream", "write_stream")
        mock_client_session.return_value.__aenter__.return_value = mock_session
        
        result = await mcp_tool.ask_question("Mixed content question")
        
        assert "Text content" in result
        assert "Image content" in result
