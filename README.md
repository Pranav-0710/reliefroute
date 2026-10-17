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
