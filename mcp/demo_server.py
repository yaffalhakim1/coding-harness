#!/usr/bin/env python3
"""A tiny MCP server used to exercise the harness's MCP client.

Implements the tool-use subset of MCP over stdio: initialize, tools/list,
tools/call. Used by `config/mcp-demo.toml` and the end-to-end tests.

Tools:
  word_count   count words in a string
  shout        upper-case a string and append "!"
"""

import json
import sys


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


TOOLS = [
    {
        "name": "word_count",
        "description": "Count the words in a piece of text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "shout",
        "description": "Convert text to upper case and append an exclamation mark.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]


def call_tool(name, args):
    text = str(args.get("text", ""))
    if name == "word_count":
        return {"content": [{"type": "text", "text": str(len(text.split()))}], "isError": False}
    if name == "shout":
        return {"content": [{"type": "text", "text": text.upper() + "!"}], "isError": False}
    return {"content": [{"type": "text", "text": "unknown tool " + name}], "isError": True}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method")
        mid = msg.get("id")

        # Notifications carry no id and expect no reply.
        if mid is None:
            continue

        if method == "initialize":
            send({
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "demo-tools", "version": "1.0"},
                },
            })
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params", {})
            result = call_tool(params.get("name"), params.get("arguments", {}))
            send({"jsonrpc": "2.0", "id": mid, "result": result})
        else:
            send({
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32601, "message": "unknown method " + str(method)},
            })


if __name__ == "__main__":
    main()
