from mcp.server.fastmcp import FastMCP
import logging
import os
from langchain_community.utilities import GoogleSerperAPIWrapper
import dotenv
dotenv.load_dotenv()
# Initialize FastMCP server
mcp = FastMCP("web-search", port=8001)

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


search_engine = GoogleSerperAPIWrapper(k=5, api_key=os.getenv("SERPER_API_KEY"))

async def search_helper(query: str):
    try:
        # Use "text" backend to avoid problematic multi-engine search (grokipedia, etc.)
        results = search_engine.run(query)
        return results
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        # Return a helpful error message instead of crashing
        return f"Search failed due to network error: {str(e)}. Please try again or check your internet connection."


@mcp.tool(
    name="search",
    description="Search the web using DuckDuckGo. Returns relevant search results for the given query string."
)
async def search(query: str) -> str:
    """
    Search the web using Google.

    Args:
        query: A search query string (e.g. "current weather in Berlin" or "Python tutorial")

    Returns:
        Search results as text snippets.
    """
    results = await search_helper(query)
    return results


def main():
    # Initialize and run the server
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()