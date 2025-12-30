import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession
from mcp.types import ListToolsResult
from openai import OpenAI
import os
import dotenv
dotenv.load_dotenv()

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolNamespaceConfig,
    ToolDescription,
    load_harmony_encoding,
)

async def list_server_and_tools(server_url: str):
    async with sse_client(url=server_url) as streams, ClientSession(
            *streams) as session:
        initialize_response = await session.initialize()
        list_tools_response = await session.list_tools()
        return initialize_response, list_tools_response

def convert_tool_format(tool):
    # MCP JSON Schema may omit "required" when nothing is required.
    # It may also omit "properties" for non-object schemas (rare, but guard anyway).
    properties = {}
    required = []
    if isinstance(tool.inputSchema, dict):
        properties = tool.inputSchema.get("properties") or {}
        required = tool.inputSchema.get("required") or []

    # `client.responses.create()` expects the "Responses API" tool format:
    # { "type": "function", "name": "...", "description": "...", "parameters": { ...json schema... } }
    converted_tool = {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
    return converted_tool
 
initialize_response, list_tools_response = asyncio.run(
    list_server_and_tools("http://127.0.0.1:8001/sse"))

tools = [convert_tool_format(tool) for tool in list_tools_response.tools]

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)
# 2. Prompt the model with tools defined
response = client.responses.create(
    model="openai/gpt-oss-120b:free",
    input="search for martin moder on the web",
    tools=tools,
)

print(response)