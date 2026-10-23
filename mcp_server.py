"""MCP server (Streamable HTTP, stateless, JSON responses) exposing ReliefRoute's agent tools.

Any MCP client can connect to POST /mcp, e.g. Gemini CLI settings.json:
    {"mcpServers": {"reliefroute": {"httpUrl": "https://<host>/mcp"}}}
or Google ADK: McpToolset(connection_params=StreamableHTTPConnectionParams(url="https://<host>/mcp")).
"""
import json

import agent
import ops

SUPPORTED_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

TOOLS = [
    {"name": "parse_field_report",
     "description": "Extract structured relief needs (zone, item, qty, urgency) and road hazards from a "
                    "free-text field report in any language.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "dispatch_relief",
     "description": "End to end: understand a field report, allocate depot stock in Chennai, dispatch deliveries, "
                    "survive a live road disruption by replanning (self-healing) and learn from the outcome. "
                    "Returns deliveries, self-heal events, score and new lessons.",
     "inputSchema": {"type": "object",
                     "properties": {"text": {"type": "string"},
                                    "chaos": {"type": "boolean", "default": True,
                                              "description": "simulate a bridge collapse mid-route"}},
                     "required": ["text"]}},
    {"name": "train_agent",
     "description": "Run simulated disaster episodes so the agent improves its policy and lesson memory.",
     "inputSchema": {"type": "object",
                     "properties": {"episodes": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1}}}},
    {"name": "get_operation",
     "description": "Live operation status: figures, waiting incidents, trucks on routes and active road hazards.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "get_memory",
     "description": "Learned lessons, per-strategy policy stats and score history.",
     "inputSchema": {"type": "object", "properties": {}}},
]
_NAMES = {t["name"] for t in TOOLS}


def _call(name, args):
    if name == "parse_field_report":
        return agent.intake(str(args.get("text", "")))
    if name == "dispatch_relief":
        heals = []
        report = agent.intake(str(args.get("text", "")), heals=heals)
        r = agent.run_episode(extra_needs=report["needs"], extra_hazards=report["hazards"],
                              chaos=bool(args.get("chaos", True)),
                              strategy="llm" if agent.llm_available() else None, use_maps=True, explore=False)
        return {"intake": report, "strategy": r["strategy"], "score_pct": r["score"],
                "total_reward": r["total_reward"], "deliveries": r["deliveries"], "unmet": r["unmet"],
                "self_heal_events": heals + r["heals"], "new_lessons": r["new_lessons"]}
    if name == "train_agent":
        episodes, _ = agent.train(max(1, min(5, int(args.get("episodes", 1)))))
        return {"episodes": episodes}
    if name == "get_operation":
        st = ops.load()
        return {"summary": ops.summary(st),
                "waiting": [{k: i[k] for k in ("id", "zone", "item", "qty", "urgency", "note")}
                            for i in st["incidents"] if i["qty"] > 0 and not i["closed"]],
                "routes": [{"id": r["id"], "truck": r["truck_id"],
                            "stops": [{k: s[k] for k in ("zone", "item", "qty", "status", "eta_min")} for s in r["stops"]]}
                           for r in st["routes"] if r["status"] == "active"],
                "hazards": [h["label"] for h in st["hazards"] if h["active"]]}
    mem = agent.load_memory()
    return {"lessons": mem["lessons"], "policy": mem["bandit"], "episodes": mem["episodes"][-20:]}


def ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def error(mid, code, message):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def handle(msg):
    """Handle one JSON-RPC message; returns the response, or None for notifications."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid Request")
    if "id" not in msg:
        return None
    mid, method, params = msg["id"], msg["method"], msg.get("params") or {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        return ok(mid, {"protocolVersion": asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0],
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "reliefroute", "version": "1.0.0"},
                        "instructions": "Flood-relief logistics agent for Chennai. Use dispatch_relief for a field "
                                        "report, train_agent to improve it, get_memory to inspect what it learned."})
    if method == "ping":
        return ok(mid, {})
    if method == "tools/list":
        return ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in _NAMES:
            return error(mid, -32602, f"Unknown tool: {name}")
        try:
            result = _call(name, args)
        except Exception as e:  # tool errors are reported to the model, not as protocol errors
            return ok(mid, {"content": [{"type": "text", "text": f"Tool failed: {e}"}], "isError": True})
        return ok(mid, {"content": [{"type": "text", "text": json.dumps(result)}], "structuredContent": result,
                        "isError": False})
    return error(mid, -32601, f"Method not found: {method}")
