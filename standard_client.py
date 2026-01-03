import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

# Keep names used by the original file, but point SERVER_CONFIG at the SSE URL.
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b:free")
SERVER_CONFIG = "http://127.0.0.1:8001/sse"
MAX_TOOL_CALLS = 5  # Limit to prevent infinite loops


def convert_tool_format(tool):
    # IMPORTANT: Chat Completions expects tools in the format:
    # { "type": "function", "function": { "name": "...", "description": "...", "parameters": { ... } } }
    #
    # MCP JSON Schema may omit "required" when nothing is required.
    # It may also omit "properties" for non-object schemas (rare, but guard anyway).
    properties = {}
    required = []
    if isinstance(tool.inputSchema, dict):
        properties = tool.inputSchema.get("properties") or {}
        required = tool.inputSchema.get("required") or []

    converted_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            # OpenAI/OpenRouter expects a string; guard against None.
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    return converted_tool


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    async def connect_to_server(self, server_config):
        # Use SSE transport (the change you asked for, like client.py:53-55).
        streams = await self.exit_stack.enter_async_context(sse_client(url=server_config))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*streams))

        await self.session.initialize()

        # List available tools from the MCP server
        response = await self.session.list_tools()
        print("\nConnected to server with tools:", [tool.name for tool in response.tools])

        self.messages = []

    async def process_query(self, query: str) -> str:

        self.messages.append({
            "role": "system",
            "content": """You are a helpful assistant with access to tools.

Rules:
1. Call tools when you need information. Use valid JSON for arguments.
2. Once you have enough information, respond with a final answer (no more tool calls).
3. Never call a tool twice with the same arguments.
4. Keep search queries short and specific (3-6 words work best)."""
        })

        self.messages.append({
            "role": "user",
            "content": query
        })

        response = await self.session.list_tools()
        available_tools = [convert_tool_format(tool) for tool in response.tools]

        response = self.openai.chat.completions.create(
            model=MODEL,
            tools=available_tools,
            messages=self.messages
        )
        self.messages.append(response.choices[0].message.model_dump())

        final_text = []
        tool_call_count = 0

        # Handle potential multiple rounds of tool calls
        while True:
            content = response.choices[0].message
            self.messages.append(content.model_dump())

            # Check if we've exceeded the max tool calls
            if tool_call_count >= MAX_TOOL_CALLS:
                final_text.append(f"[Stopped: exceeded maximum of {MAX_TOOL_CALLS} tool call rounds]")
                # Force a final response without tools
                self.messages.append({
                    "role": "user",
                    "content": "Please provide your final answer based on the information gathered so far. Do not make any more tool calls."
                })
                final_response = self.openai.chat.completions.create(
                    model=MODEL,
                    messages=self.messages,
                    max_tokens=1000,
                    # No tools = forces text response
                )
                final_text.append(final_response.choices[0].message.content or "")
                break

            if content.tool_calls is not None:
                tool_call_count += 1
                for tool_call in content.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_raw = tool_call.function.arguments
                    try:
                        tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse tool arguments: {e}")
                        print(f"Raw arguments: {repr(tool_args_raw)}")
                        # Send error back to model so it can retry
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: Invalid JSON in tool arguments. Please try again with valid JSON. Details: {e}"
                        })
                        continue

                    # Execute tool call
                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                        tool_content = result.content
                    except Exception as e:
                        print(f"Error calling tool {tool_name}: {e}")
                        tool_content = f"Error executing tool: {e}"

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content
                    })

                # Continue with another completion after tool results
                response = self.openai.chat.completions.create(
                    model=MODEL,
                    tools=available_tools,
                    max_tokens=1000,
                    messages=self.messages,
                )
            else:
                # No more tool calls, this is the final response
                if content.content == "":
                    print("No content returned from model")
                final_text.append(content.content)
                break

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                result = await self.process_query(query)
                print("Result:")
                print(result)

            except Exception as e:
                print(f"Error: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server(SERVER_CONFIG)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
