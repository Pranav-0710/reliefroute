# ReliefRoute

ReliefRoute is a flood-relief logistics agent for Chennai. It takes a field report (typed text, a photo, or a voice note, in English, Tamil, Tanglish or Hindi) and handles the rest:
- it works out who needs what and how urgently
- it assigns supplies from depots and dispatches deliveries
- it detects when a route fails mid-delivery and replans
- it scores the outcome and learns from it before the next run

Built for the Agent Wars challenge at GDG DevFest 2026.

| | |
|---|---|
| Live app | https://reliefroute.vercel.app |
| MCP endpoint | https://reliefroute.vercel.app/mcp |
| Health check | https://reliefroute.vercel.app/healthz |
| Source | https://github.com/Pranav-0710/reliefroute |
| Architecture details | [docs/architecture.md](docs/architecture.md) |

![ReliefRoute operations console](docs/screenshot.png)

---

## The problem

During the Chennai floods, help requests arrive as WhatsApp voice notes, photos of handwritten lists and messages in mixed Tamil and English. Coordinators have to work out by hand:
- what each message is asking for
- which depot has the stock
- which roads are still open

Roads close while trucks are already moving, and the lessons from one flood are rarely written down before the next one.

ReliefRoute automates that chain from the moment a report arrives to a scored, completed delivery plan.

## What the agent does

Every dispatch runs the same seven stages. The web app lights them up as they complete.

```mermaid
flowchart LR
    A["Field report<br/>text, photo, voice"] --> B["1 Understand<br/>Gemini multimodal"]
    B --> C["2 Plan<br/>strategy + lessons"]
    C --> D["3 Dispatch<br/>simulator step"]
    D --> E{"4 Disruption?<br/>route blocked"}
    E -->|yes| F["5 Self-heal<br/>replan or detour"]
    F --> D
    E -->|no| G["6 Score"]
    G --> H["7 Learn<br/>lessons + policy"]
    H -.->|next episode| C
```

| Stage | What happens |
|---|---|
| 1 Understand | Gemini reads the report and any attached photo or audio. It returns structured needs (area, item, quantity, urgency 1–5) and road hazards. The agent validates the output: it clamps quantities and checks that coordinates fall inside Chennai. |
| 2 Plan | The agent picks a planning strategy (Gemini or one of three rule-based strategies) and assigns each request to a depot. A guardrail checks every plan Gemini produces before anything moves. |
| 3 Dispatch | The simulator executes the plan: it moves stock and computes distance, travel time and reward. |
| 4 Disruption | On the first step, a bridge collapses on the longest route that is not already detouring. That delivery fails. |
| 5 Self-heal | The agent sees which deliveries failed and replans them: a detour or a different depot. It gets up to three rounds. |
| 6 Score | The reward is converted to a percentage of the best possible outcome. |
| 7 Learn | The strategy's average score is updated, and the agent writes lessons that are fed into the next plan. |

## Try it in three minutes

1. Open https://reliefroute.vercel.app.
2. Under **Training**, click **Run 5 episodes**. Click it a second time.
   - The agent plays ten random flood scenarios.
   - Watch the **Performance** panel: score per episode, the change against the first five runs, and which strategy the policy now prefers.
3. Under **Field report**, click the **Tanglish** sample (or attach a photo, or record a voice note), then click **Dispatch**.
4. Read the result:
   - **Map:** orange dots are your report and blue dots are other open requests. A red cross is the route that broke; a dashed green line is the detour that replaced it.
   - **Event log:** each line is tagged with the stage that produced it.

| Log tag | Meaning |
|---|---|
| `INTAKE` | What Gemini extracted from the report |
| `STEP n` | The plan for that round, its reasoning, and each delivery |
| `ALERT` | A disruption happened (the bridge collapse) |
| `REROUTE` | A blocked delivery is being replanned |
| `GUARDRAIL` | The plan was corrected before dispatch (unknown depot, missing request, unsafe route) |
| `FAILOVER` | A Gemini model ran out of quota, so the agent switched model |
| `FALLBACK` | An external service failed, so the agent used a local method (rules, distance estimate) |
| `SCORE` | Outcome as a percentage of the best possible |
| `LESSON` | A lesson written to memory for the next run |

## Architecture

```mermaid
flowchart LR
    subgraph Client["Browser"]
        UI["static/index.html<br/>report form, map, event log, charts"]
    end
    EXT["Other agents<br/>Gemini CLI, Google ADK"]
    subgraph Server["FastAPI app on Vercel"]
        API["main.py<br/>REST API"]
        MCP["mcp_server.py<br/>MCP endpoint"]
        AG["agent.py<br/>intake, planning, guardrails,<br/>failover, reflection, bandit"]
        SIM["env.py<br/>flood simulator<br/>reset / step / state"]
        MEM[("Memory<br/>lessons, policy stats, scores")]
    end
    subgraph Google["Google APIs"]
        GEM["Gemini API"]
        ROUTES["Maps Routes API"]
    end
    UI -->|HTTPS| API
    EXT -->|JSON-RPC| MCP
    API --> AG
    MCP --> AG
    AG <--> SIM
    AG <--> MEM
    AG --> GEM
    AG --> ROUTES
```

- **Agent (`agent.py`)** holds all the decision logic. Both the web API and the MCP server call the same functions.
- **Simulator (`env.py`)** follows the OpenEnv contract (`reset`, `step`, `state`). It runs in-process, so an episode with the rule-based strategies takes milliseconds.
- **Memory** is a JSON document: lessons, per-strategy statistics and episode history. The browser keeps a copy and sends it back with each request. If a serverless instance restarts and loses its copy, learning is restored from the browser.
- **External calls go only to Google APIs:** Gemini for understanding, planning and reflection, and the Maps Routes API for travel times.

The request lifecycle, simulator rules, scoring formula, learning loop and every failure path are documented with diagrams in [docs/architecture.md](docs/architecture.md).

## How it meets the brief

| Requirement | How ReliefRoute does it | Where |
|---|---|---|
| End to end | One request goes from raw report to understood needs, plan, dispatch, recovery, score and lesson, with no human step | `agent.run_episode` |
| Multimodal | Text, images and audio go to Gemini in a single call. The browser converts microphone recordings to WAV. | `agent.intake`, `static/index.html` |
| Feedback loop | Every episode's score updates the policy and the lesson memory, and both shape the next episode | `agent.run_episode`, `agent.reflect` |
| Self-learning | 1. An epsilon-greedy bandit learns which planning strategy scores best. 2. Gemini writes lessons that are injected into later planning prompts. | `agent.select_strategy`, `agent.llm_plan` |
| Self-healing | Replanning after blocked routes; guardrail repair of bad plans; retries; a quota circuit breaker with model failover; a rule-based fallback for every Gemini step; distance estimates when Maps fails; memory restore after cold starts | `agent.repair`, `agent.gemini_json`, `agent.maps_eta`, `agent.sync_memory` |
| Google APIs only | Gemini API and Google Maps Routes API, called with Google API keys | `agent.py` |
| Deployable | Live on Vercel. A Dockerfile is included for Cloud Run. | `Dockerfile` |
| MCP | The agent's tools are exposed over MCP (Streamable HTTP) | `mcp_server.py` |

## Run it locally

Requirements: Python 3.10 or newer. A Gemini API key from https://aistudio.google.com/apikey is optional; without one, the agent runs entirely on its rule-based fallbacks.

```bash
git clone https://github.com/Pranav-0710/reliefroute.git
cd reliefroute
python -m venv .venv
```

Activate the environment. On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY = "your-key"
```

On macOS or Linux:

```bash
source .venv/bin/activate
export GEMINI_API_KEY="your-key"
```

Install the dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn main:app --port 8080
```

Open http://localhost:8080.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | no | – | Enables Gemini. `GOOGLE_API_KEY` also works. |
| `MAPS_API_KEY` | no | the Gemini key | Key for the Maps Routes API |
| `GEMINI_MODELS` | no | `gemini-flash-latest,gemini-3.8-flash,gemini-3.5-flash-lite` | Order in which models are tried |
| `MEMORY_PATH` | no | `memory.json` (`/tmp/memory.json` on Vercel) | Where memory is stored |
| `OFFLINE` | no | – | Set to `1` to force rule-based mode, even when a key is present |

### Tests

```bash
pytest -q
```

The 11 tests run without network access or an API key. They cover:
- the simulator contract
- recovery after a blocked route
- guardrail repair
- the bandit learning to avoid the weakest strategy
- scores staying within 0–100% across 120 episodes
- memory restore after a cold start
- the quota circuit breaker
- keyword intake
- the MCP handshake and tool calls
- the full web flow

## Use it from another agent (MCP)

The MCP endpoint is stateless Streamable HTTP with JSON responses.

| Tool | Input | Returns |
|---|---|---|
| `parse_field_report` | `text` | Structured needs and hazards |
| `dispatch_relief` | `text`, optional `chaos` | Deliveries, self-heal events, score and new lessons |
| `train_agent` | optional `episodes` (1–5) | Strategy and score for each episode |
| `get_memory` | – | Lessons, policy statistics and recent scores |

List the tools:

```bash
curl -s https://reliefroute.vercel.app/mcp -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

Call a tool:

```bash
curl -s https://reliefroute.vercel.app/mcp -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"dispatch_relief","arguments":{"text":"Adyar: 200 people need drinking water, urgent"}}}'
```

Gemini CLI (`~/.gemini/settings.json`):

```json
{ "mcpServers": { "reliefroute": { "httpUrl": "https://reliefroute.vercel.app/mcp" } } }
```

Google ADK:

```python
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

reliefroute = McpToolset(connection_params=StreamableHTTPConnectionParams(url="https://reliefroute.vercel.app/mcp"))
```

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Web app |
| `GET` | `/healthz` | Status, whether Gemini is configured, model order |
| `POST` | `/api/dispatch` | Multipart form: `text`, `chaos`, optional `image`, `audio`, `memory`. Runs one full episode for the report. |
| `POST` | `/api/train` | JSON `{"episodes": 1-20, "chaos": true}`. Runs simulated episodes. |
| `GET` | `/api/scenario` | A sample scenario for the initial map |
| `GET` | `/api/memory` | Current memory |
| `POST` | `/api/memory/reset` | Clear memory |
| `POST` | `/mcp` | MCP JSON-RPC endpoint |

## Repository layout

| Path | Contents |
|---|---|
| `env.py` | Flood simulator: depots, requests, floods, the bridge collapse, reward |
| `agent.py` | Intake, planning strategies, guardrail, Gemini failover, Maps ETAs, reflection, bandit, memory |
| `main.py` | FastAPI server: web app, REST API, MCP route |
| `mcp_server.py` | MCP protocol handling and tool definitions |
| `static/index.html` | The operations console (single file, no build step) |
| `test_app.py` | Automated tests |
| `Dockerfile` | Container image for Cloud Run or any container host |
| `docs/architecture.md` | Diagrams and technical reference |

## Deployment

The live app runs on Vercel, which detects the FastAPI app in `main.py` automatically:

```bash
npx vercel deploy --prod
```

Set `GEMINI_API_KEY` under the project's Environment Variables first. The live deployment sets `GEMINI_MODELS=gemini-3.5-flash-lite,gemini-flash-latest,gemini-3.8-flash`, which starts with the model that has the most free-tier quota.

To run on Google Cloud Run instead, use the included Dockerfile. This needs a project with billing enabled:

```bash
gcloud run deploy reliefroute --source . --region asia-south1 --allow-unauthenticated --set-env-vars GEMINI_API_KEY=your-key
```

## Known limitations

- **Google Maps travel times are not live yet.** The Routes API call is implemented, but the deployed key is not authorised for that API (it returns `401`). Enabling the Routes API requires a Google Cloud project with billing, and ours is closed. The agent falls back to distance estimates, and the log labels those ETAs "estimate". Setting `MAPS_API_KEY` to a key with the Routes API enabled switches to real travel times with no code change.
- **Gemini quota is limited.** The deployment uses the Gemini free tier, which allows few requests per day. When a model is exhausted, the circuit breaker moves to the next one. If every model is exhausted, the agent continues on rule-based intake and planning, which only understands English keywords and known area names.
- **The scenario data is simulated.** Depot stock, the other requests, flood zones and the bridge collapse are all simulated. Locations are real Chennai localities with approximate coordinates. Distances are straight-line multiplied by a 1.3 road factor, not a road network.
- **Memory is not durable.** It is a JSON file on the serverless instance, plus a copy in each browser. It is not reliably shared between instances or users. A production version would store it in Firestore.
- **Learning does not train model weights.** It is a bandit over four strategies, plus lessons added to Gemini prompts. The simulator follows the OpenEnv `reset`/`step`/`state` contract but is not yet packaged as an OpenEnv server for RL training.
- **There is no authentication.** Anyone with the link can run dispatches and reset memory. The MCP server is minimal: tools only, no sessions or streaming.

## Next steps

- Package `env.py` as an OpenEnv server and train a small planning model with GRPO (Hugging Face TRL) against it.
- Store memory in Firestore so learning is shared and durable.
- Use the Routes API route matrix for real road distances, and the Route Optimization API for multi-stop trucks.
- Connect real depot inventory and a phone or SMS intake line.
- Add authentication and per-organisation memory.

## Author

Built by jey pranav ([@Pranav-0710](https://github.com/Pranav-0710)).

## License

MIT. See [LICENSE](LICENSE).
