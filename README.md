---
title: ReliefRoute
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8080
pinned: false
---

# ReliefRoute

Multimodal, self-healing, self-learning flood-relief logistics agent built on Gemini and Google Maps.

**Flow:** field report (text / photo / voice, any language) → Gemini extracts needs and hazards → allocate depot stock →
plan routes → live disruption (bridge collapse) → agent detects the blocked delivery and reroutes → episode reward →
Gemini reflection writes lessons + bandit policy update → next run is better.

- `env.py`: OpenEnv-style simulator (`reset` / `step` / `state`) generating random flood scenarios
- `agent.py`: intake, planning, guardrails, retries/model fallback, Routes API ETAs, reflection, bandit learning
- `main.py`: FastAPI API + web UI (`static/index.html`)
- `test_app.py`: offline checks (`pytest`)

## Run

```bash
pip install -r requirements.txt
GEMINI_API_KEY=... uvicorn main:app --port 8080
```

Env vars: `GEMINI_API_KEY` (required for AI), `MAPS_API_KEY` (optional, Routes API), `GEMINI_MODELS` (fallback order).

## MCP

The agent's tools are exposed as an MCP server (Streamable HTTP) at `/mcp`:
`parse_field_report`, `dispatch_relief`, `train_agent`, `get_memory`.

Gemini CLI (`~/.gemini/settings.json`):

```json
{"mcpServers": {"reliefroute": {"httpUrl": "https://reliefroute.vercel.app/mcp"}}}
```

Google ADK:

```python
McpToolset(connection_params=StreamableHTTPConnectionParams(url="https://reliefroute.vercel.app/mcp"))
```
