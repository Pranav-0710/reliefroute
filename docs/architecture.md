# ReliefRoute architecture

This document explains how ReliefRoute works internally. For setup and a quick tour, see the [README](../README.md).

1. [Modules](#1-modules)
2. [Lifecycle of a dispatch](#2-lifecycle-of-a-dispatch)
3. [The simulator](#3-the-simulator)
4. [Reward and score](#4-reward-and-score)
5. [Planning strategies and the guardrail](#5-planning-strategies-and-the-guardrail)
6. [The learning loop](#6-the-learning-loop)
7. [Self-healing](#7-self-healing)
8. [Multimodal intake](#8-multimodal-intake)
9. [MCP server](#9-mcp-server)
10. [Deployment and state](#10-deployment-and-state)
11. [Tests](#11-tests)
12. [Live operations](#12-live-operations)

---

## 1. Modules

```mermaid
flowchart TD
    page["static/index.html<br/>operations console"] -.->|fetch| main["main.py<br/>FastAPI routes"]
    ext["MCP clients"] -.->|JSON-RPC| main
    main --> agent["agent.py<br/>decision logic"]
    main --> mcp["mcp_server.py<br/>MCP protocol + tools"]
    mcp --> agent
    agent --> env["env.py<br/>flood simulator"]
    agent --> gemini["Gemini API<br/>via google-genai"]
    agent --> routes["Maps Routes API<br/>via httpx"]
    agent --> mem[("memory.json")]
```

| Module | Responsibility |
|---|---|
| `env.py` | Builds a flood scenario, executes delivery plans, injects the disruption, computes the reward. Pure Python, no I/O. |
| `agent.py` | Everything the agent decides: intake, strategy choice, planning, plan repair, model failover, travel times, reflection, memory. |
| `main.py` | HTTP layer. Turns requests into calls on `agent.py` and returns JSON. |
| `mcp_server.py` | Implements the MCP methods and maps four tools onto `agent.py`. |
| `static/index.html` | The console: report form, pipeline, map, event log, performance panel. One file, no build step. |
| `policy.py` | Features and decision procedure of the learned policy, plus inference in plain Python from `policy_weights.json`. |
| `rl/train_policy.py` | Trains the policy with REINFORCE in PyTorch. Only needed offline. |
| `relief_openenv/` | The simulator wrapped as an OpenEnv environment (typed models, server, client, manifest). |
| `ops.py` | Live operations: incident queue, dispatch waves, restore from the browser, situation reports. |
| `store.py` | Shared state storage: Cloud Firestore (REST, conditional writes with retry) or local JSON files. |
| `static/report.html` | Field reporter phone page. |

## 2. Lifecycle of a dispatch

What happens when someone submits a report on the web app (`POST /api/dispatch`):

```mermaid
sequenceDiagram
    participant U as Browser
    participant API as main.py
    participant AG as agent.py
    participant G as Gemini API
    participant SIM as env.py
    participant M as Routes API
    U->>API: report text, photo, audio, memory copy
    API->>AG: sync_memory, then intake
    AG->>G: prompt + photo + audio
    G-->>AG: JSON needs and hazards
    AG->>AG: validate and clamp fields
    AG->>SIM: reset with field needs and hazards
    loop up to 3 steps, until every need is served
        AG->>G: observation + stored lessons
        G-->>AG: assignments + reasoning
        AG->>AG: guardrail repair
        AG->>SIM: step(assignments)
        SIM-->>AG: deliveries, blocked routes, reward
    end
    AG->>G: episode summary
    G-->>AG: 1 or 2 lessons
    AG->>M: travel time for each field delivery
    M-->>AG: duration, or error then estimate
    AG->>AG: update policy stats and memory
    API-->>U: trace, self-heal events, score, lessons, memory
```

If any Gemini call fails, that step falls back to a local method instead of failing the request (see [section 7](#7-self-healing)). With the free-tier model, a typical dispatch takes 6–10 seconds.

## 3. The simulator

`env.ReliefEnv` follows the OpenEnv environment contract, so the same environment can later back RL training.

| Method | Returns |
|---|---|
| `reset(seed, extra_needs, extra_hazards, chaos, n_needs=5)` | Initial observation |
| `step(action)` | `{"observation", "reward", "done", "info"}` |
| `state()` | Full episode state, including history and total reward |

**Action**

```json
{"assignments": [{"need_id": "R1", "depot_id": "D2", "detour": false, "truck_id": "D2-T1"}]}
```

The order of the list is the loading order. Each truck visits its stops in the order they were assigned to it. `truck_id` is optional: without it, the depot's truck that is free earliest takes the stop.

**Observation**

| Field | Contents |
|---|---|
| `depots` | Id, name, coordinates, current stock per item, truck ids |
| `truck_capacity` | Units one truck can carry in a round |
| `pending_needs` | Unserved requests: zone, coordinates, item, remaining quantity, urgency 1–5, source (`sim` or `field`) |
| `known_hazards` | Flood zones and collapses reported so far |
| `route_matrix` | For every pending need and depot: distance in km, and which hazards the route crosses |
| `step`, `max_steps` | Progress through the episode |

**World parameters**

| Parameter | Value |
|---|---|
| Depots | 3 real locations: Guindy, Anna Nagar, Tondiarpet |
| Stock | 60–260 units of each of water, food, medicine, blankets, per depot |
| Simulated requests | 5 per episode, in 10 Chennai localities, 30–150 units, urgency 1–5 |
| Field requests | Added from the report, ids `R1`, `R2`, … |
| Fleet | 2 trucks per depot, 250 units each per round. Trucks start each round empty at their depot. |
| Flood zones | 1–2 random, radius 1.0 km. Hazards from the report have radius 0.8 km. |
| Disruption | On step 1, a bridge collapse (radius 0.6 km) at the midpoint of the longest leg the trucks are about to drive without a detour |
| Distance | Straight line in a local projection, multiplied by 1.3 for road winding |
| Detour | Distance multiplied by 1.6. A detour always gets through. |
| Speed | 25 km/h |
| Episode length | Ends when every need is served, or after 3 steps |

**Rules applied in each step (one dispatch round)**
- Invalid assignments are ignored and counted in `info.invalid`: unknown ids, needs already served, and duplicate assignments for the same need.
- Each assignment is loaded onto a truck of its depot. If every truck at that depot is full, it goes to `info.no_truck` and the need stays pending.
- A truck drives **leg by leg**: depot → first stop → second stop, and so on. Each leg is checked against every hazard circle. A blocked leg without `detour` fails; the truck stays where it is and carries on to its next stop.
- The amount delivered is the smallest of: what the need still requires, the depot's stock, and the truck's remaining capacity. The rest of the need stays pending for the next round.
- Arrival time accumulates along the route, so the third stop of a truck arrives later than its first.

```mermaid
flowchart LR
    D["Depot D1"] -->|"leg 1"| S1["Stop 1<br/>urgent medicine"]
    S1 -->|"leg 2"| S2["Stop 2<br/>food"]
    S2 -->|"leg 3, blocked"| S3["Stop 3<br/>stays pending"]
    S2 -.->|"next round"| S3
```

## 4. Reward and score

For each step:

```
delivered   +  urgency × 10 × delivered_qty / requested_qty
            −  urgency × arrival_hours × 4      (arrival_hours = km driven by that truck so far / 25)
blocked     −  5
no stock    −  2
no truck    −  2
```

When the episode ends, every request still not fully served costs:

```
−  urgency × 10 × undelivered_qty / requested_qty
```

The score shown in the app normalises this against the best case, where every request is fully served with zero travel time:

```
score % = max(0, 100 × total_reward / Σ(urgency × 10))
```

Normalising makes easy and hard episodes comparable, which is what the learning loop needs.

## 5. Planning strategies and the guardrail

| Strategy | Order | Depot choice | Floods |
|---|---|---|---|
| `llm` | Gemini decides | Gemini decides, from the observation and stored lessons | Gemini is told to detour around `blocked_by` routes, and the guardrail enforces it |
| `rl_policy` | Learned (need network) | Learned (depot network); pins the depot's earliest-free truck | Detours when the actual leg (previous stop to this stop) crosses a known hazard |
| `urgency_first` | Most urgent first | Nearest depot with enough stock | Detours around known hazards |
| `nearest_first` | Arrival order | Nearest depot with enough stock | Detours around known hazards |
| `fifo` | Arrival order | First depot with any stock | Ignores hazards |

`fifo` is a deliberately naive baseline. It gives the learning loop a clearly bad option to learn to avoid.

**Guardrail** (`agent.repair`). Every Gemini plan goes through these checks before it reaches the simulator:
1. Assignments with an unknown need or depot, or a duplicate need, are dropped.
2. Assignments whose route crosses a known hazard without a detour are switched to a detour.
3. Pending needs that the plan left out are filled in with `urgency_first`.

Each correction is logged as a `GUARDRAIL` event.

## 6. The learning loop

```mermaid
flowchart TD
    A["Choose strategy<br/>bandit: best average, or explore"] --> B["Plan<br/>Gemini reads stored lessons"]
    B --> C["Simulator runs the episode<br/>deliveries, blocks, reward"]
    C --> D["Score %"]
    D --> E["Update the strategy's<br/>running average"]
    D --> F["Reflect<br/>write 1-2 lessons"]
    E --> MEM[("Memory")]
    F --> MEM
    MEM --> A
    MEM --> B
```

**Offline RL: the `rl_policy` strategy** (`policy.py`, `rl/train_policy.py`)

```mermaid
flowchart LR
    S["Simulator episode<br/>random seed"] --> F["Features per request<br/>and per depot"]
    F --> N1["Need network<br/>pick next request"]
    N1 --> N2["Depot network<br/>pick depot"]
    N2 --> A["Round of assignments"]
    A --> S
    S -->|"episode return"| R["REINFORCE update<br/>advantage = return − batch mean"]
    R --> N1
    R --> N2
```

- Each network is one tanh hidden layer of 24 units. The need network scores every pending request, and sampling from its softmax picks the next one to load (this sets the stop order). The depot network then scores the depots that have both stock and a truck with space.
- While deciding, `policy.decide` simulates the fleet the same way the simulator loads it (`_fleet`, `_truck_for`). Features use the real arrival time, the actual leg and whether it crosses a known hazard, and the truck's remaining space. The resulting plan pins each stop to that truck (`truck_id`), so what the policy imagined is exactly what gets driven.
- `policy.decide` is written without any framework, so training (PyTorch tensors, sampled choices) and serving (plain floats, greedy choices) run exactly the same procedure.
- Training: 32,000 episodes, Adam with learning rate 0.005, return = reward / best possible reward. The baseline is **group-relative** (as in GRPO): each scenario is played 8 times per update, and an attempt's advantage is its return minus the mean return of its group, normalised over the batch.
- Evaluation: 500 scenarios with seeds disjoint from training, all with the bridge collapse on. Results are in `rl/results.json`: 78.0% untrained, 80.6% trained, against 80.0% for `urgency_first`.
- Paired over the same 500 scenarios: +0.57 points (95% CI +0.33 to +0.81), with 155 wins, 238 ties and 107 losses.
- Best-of-40 sampled plans with hindsight reach about 82.5% on 100 of these scenarios, which is an estimate of the remaining headroom.

**Online policy selection: an epsilon-greedy bandit** (`agent.select_strategy`)
- Strategies that have never been tried are tried first.
- After that, 80% of the time it picks the strategy with the best average score; 20% of the time it picks one at random, so a strategy that has improved can be rediscovered.
- After each episode, that strategy's average is updated incrementally with the new score.
- `llm` is only an option when a Gemini key is configured, and `rl_policy` only when `policy_weights.json` is present.
- Training runs use the bandit. Dispatches from the web app always use `llm` when available, because only Gemini can read photos, audio and non-English text.

**Reflection** (`agent.reflect`)
- After an `llm` episode, Gemini receives the episode summary (rewards, blocked routes, stockouts, unmet needs, self-heal events) plus the existing lessons. It writes one or two new, concrete lessons.
- After a rule-based episode, lessons are written by rules: blocked deliveries lead to a lesson about checking hazards, unmet needs lead to a lesson about urgency.
- Lessons are deduplicated. The latest 8 are kept and inserted into every later Gemini planning prompt.

**Memory format**

```json
{
  "lessons": ["Avoid single-bridge corridors when alternative paths exist."],
  "episodes": [{"n": 1, "strategy": "urgency_first", "reward": 112.1, "score": 88.6, "heals": 2, "ts": 1792224867}],
  "bandit": {"llm": {"n": 1, "mean": 73.4}, "urgency_first": {"n": 4, "mean": 88.6},
             "nearest_first": {"n": 1, "mean": 76.8}, "fifo": {"n": 1, "mean": 19.9}}
}
```

The latest 200 episodes are kept.

## 7. Self-healing

| Failure | How it is detected | Recovery | Log tag |
|---|---|---|---|
| Route blocked mid-delivery | `info.blocked` from the simulator | Need stays pending, then is replanned next step (detour or another depot) | `ALERT`, `REROUTE` |
| No truck capacity left at a depot | `info.no_truck` from the simulator | Need stays pending and is loaded in the next round | `STEP n` |
| Gemini plan is invalid or unsafe | Guardrail checks against the observation | Drop, force a detour, or fill the gap with `urgency_first` | `GUARDRAIL` |
| Gemini model out of quota (HTTP 429) | Error text `RESOURCE_EXHAUSTED` | Circuit breaker skips that model for 5 minutes, then the next model is tried immediately | `FAILOVER` |
| Gemini transient error or bad JSON | Exception from the call or the parser | Retry once with backoff, then the next model | `FAILOVER` |
| Every Gemini model fails | All attempts raise | Intake: keyword rules. Planning: `urgency_first`. Reflection: rule-based lessons. | `FALLBACK` |
| Routes API fails | HTTP error or timeout (8 s) | Estimate from distance at 25 km/h | `FALLBACK` |
| Serverless instance lost its memory | Browser copy has more episodes than the server's | Server adopts the browser copy after sanitising it | – |

```mermaid
flowchart LR
    R["Gemini call"] --> M1["Model 1"]
    M1 -->|"429: breaker opens"| M2["Model 2"]
    M2 -->|"429: breaker opens"| M3["Model 3"]
    M1 -->|"other error: retry once"| M1
    M3 -->|"all failed"| RB["Rule-based fallback"]
```

The model order comes from `GEMINI_MODELS`. The live deployment uses `gemini-3.5-flash-lite`, then `gemini-flash-latest`, then `gemini-3.8-flash`. Requests ask for JSON output (`response_mime_type: application/json`) at temperature 0.2.

## 8. Multimodal intake

`agent.intake` sends the report text, plus any photo and audio as inline parts, to Gemini in one request. Gemini returns:

```json
{"summary": "...", "language": "Tamil-English (Tanglish)",
 "needs": [{"zone": "Velachery", "lat": 12.98, "lng": 80.22, "item": "water", "qty": 120, "urgency": 5, "evidence": "..."}],
 "hazards": [{"label": "road flooded", "lat": 13.02, "lng": 80.22}]}
```

**Validation applied to Gemini's output**
- `item` is mapped onto one of water, food, medicine, blankets.
- Coordinates must fall inside the Chennai area (lat 12.8–13.35, lng 80.0–80.4). Otherwise the locality name is looked up, and if that fails the city centre is used.
- `qty` is clamped to 1–1000 and `urgency` to 1–5, with defaults when they are missing.
- Hazards without numeric coordinates are discarded.

**Keyword fallback** (used when Gemini is unavailable)
- Items are matched on whole words, including common Tamil and Hindi terms: *thanni*, *kudineer*, *paani*, *khana*, *saapadu*, *marundhu*, *dawai*, *kambal*, *porvai*. Every item mentioned becomes its own need.
- The first number in the text is the quantity.
- The locality is matched against the known area names, with T Nagar as the default.
- Words such as *urgent*, *trapped*, *stranded* or *children* set urgency to 5. Otherwise urgency is 3.

**Voice recordings.** Browsers record WebM or Ogg, so the console converts recordings to 16 kHz mono WAV before upload. Uploaded audio files are sent as they are.

## 9. MCP server

`POST /mcp` implements the MCP Streamable HTTP transport in stateless mode, with JSON responses.

| Method | Behaviour |
|---|---|
| `initialize` | Echoes the client's protocol version if supported (`2025-06-18`, `2025-03-26`, `2024-11-05`), otherwise answers with the latest. Advertises the `tools` capability. |
| `ping` | Empty result |
| `tools/list` | The four tools with JSON Schema inputs |
| `tools/call` | Runs the tool. Returns `content` (JSON as text) and `structuredContent`. |
| Notifications | Accepted with HTTP 202 and no body |
| Batches | JSON arrays are processed message by message |

Errors use standard JSON-RPC codes:
- `-32700`: the body is not valid JSON
- `-32600`: the message is not a valid request
- `-32601`: unknown method
- `-32602`: unknown tool

Exceptions inside a tool come back as a normal result with `isError: true`, so the calling model can see what went wrong. `GET /mcp` returns 405, because the server does not open server-initiated streams.

## 10. Deployment and state

**Live hosting.** The app runs on Vercel as a serverless Python function (FastAPI zero-config). Instances start on demand and can be recycled, which affects state:
- Memory is written to `/tmp/memory.json` on the instance.
- The browser keeps a copy in `localStorage` and sends it with every dispatch and training request. If an instance restarts empty, `agent.sync_memory` restores the browser's copy, as long as it has more episodes.
- Training sends one episode per request, so no single request runs long and the UI can show progress.

**Cloud Run.** `Dockerfile` builds a container that serves the same app on `$PORT`. It is ready to deploy to Google Cloud Run once a billing-enabled project is available.

**Secrets.** The Gemini key is stored as an encrypted Vercel environment variable. It is never in the repository.

**Protecting a public deployment.** The app has no login, so two limits keep one visitor from using up the shared Gemini quota or wiping memory:

| Limit | Value | Where |
|---|---|---|
| Dispatches per client address | 8 per 10 minutes | `main.check_rate` |
| Training requests per client address | 60 per 10 minutes | `main.check_rate` |
| Costly MCP tool calls (`dispatch_relief`, `parse_field_report`, `train_agent`) | 10 per 10 minutes | `main.mcp` |
| Memory resets | 5 per 10 minutes | `main.check_rate` |
| Gemini calls per server instance per UTC day | `GEMINI_DAILY_BUDGET`, default 300 | `agent._spend_budget` |

When a rate limit is hit, the response is HTTP 429 with `Retry-After`. MCP gets a JSON-RPC error `-32000`. When the daily budget is spent, every Gemini step falls back to its rule-based method, exactly as if the API were down. All counters are in memory per instance, so they are a guard, not an exact global quota.

## 11. Tests

`test_app.py` runs with `OFFLINE=1` and a temporary memory file. No network access or key is needed.

| Test | What it proves |
|---|---|
| `test_env_contract_reset_step_state` | The simulator follows the reset / step / state contract |
| `test_invalid_actions_are_ignored_not_crashing` | Malformed actions are rejected, not fatal |
| `test_chaos_blocks_a_route_and_agent_heals` | The disruption blocks a route, and that need is delivered in a later step |
| `test_guardrail_repairs_bad_llm_plan` | Invalid Gemini assignments are dropped and the gap is filled |
| `test_learning_updates_memory_and_bandit_prefers_better_policy` | After 12 episodes there are lessons, and the policy does not prefer `fifo` |
| `test_score_is_percentage_of_best_case` | Scores stay within 0–100% across 120 episodes, including split deliveries |
| `test_memory_restored_from_browser_copy_after_cold_start` | Memory survives an instance reset, and malformed copies are ignored |
| `test_quota_circuit_breaker_skips_exhausted_model` | A 429 opens the breaker. The next call skips that model without retrying. |
| `test_mcp_protocol_and_tools` | Handshake, notifications, tool listing, a tool call and error codes |
| `test_offline_intake_extracts_need` | Keyword intake handles English and Tanglish, and does not read "underwater" as a water request |
| `test_api_end_to_end` | Web page, dispatch with an image upload, training, scenario and memory reset |
| `test_truck_capacity_caps_a_single_delivery` | One truck never carries more than its capacity; the rest waits for the next round |
| `test_multi_stop_route_is_driven_in_order` | Stops are numbered in loading order, arrivals are non-decreasing, each leg starts at the previous stop |
| `test_no_truck_when_depot_fleet_is_full` | Assignments beyond the depot's fleet are reported as `no_truck` |
| `test_policy_decide_serves_every_stocked_need` | The policy's decision procedure produces a valid plan covering every request that has stock |
| `test_trained_policy_runs_without_pytorch` | The exported weights drive a full episode with no PyTorch installed |
| `test_openenv_environment_and_server` | The OpenEnv environment plays a full episode, and its server answers `/health` and `/schema` (skipped without `openenv`) |
| `test_rate_limits_per_client` | REST and MCP limits trigger per client address. Cheap MCP methods are never limited. |
| `test_daily_gemini_budget_degrades_to_rules` | The daily Gemini budget refuses calls once spent |
| `test_firestore_writes_never_lose_a_concurrent_update` | Against a fake Firestore with real preconditions: a write that races another instance is rejected and retried, and both changes survive |
| `test_operations_and_memory_run_on_firestore` | A report, a wave and a training episode all persist through the Firestore backend |
| `test_operations_queue_wave_and_sitrep` | Phone report placed by GPS, a wave reducing stock by exactly what was delivered, delivered statuses with notes, rule-based situation report, reset, empty-queue 409 |
| `test_sitrep_stock_alerts_are_never_left_to_gemini` | Stock alerts come from the exact rule even when Gemini writes the report |
| `test_operation_keeps_collapsed_roads_and_restores_from_browser` | A wave's bridge collapse persists, the operation is restored from the browser copy after a reset, and junk in that copy is filtered |

## 12. Live operations

`ops.py` turns the agent from "one report, one episode" into a running operation.

```mermaid
sequenceDiagram
    participant F as Phone page
    participant C as Console
    participant API as main.py
    participant O as ops.py
    participant AG as agent.py
    F->>API: POST /api/reports (text, photo, voice, GPS)
    API->>AG: intake
    API->>O: add_report: incidents I1, I2 waiting
    C->>API: GET /api/ops every 5 s
    API-->>C: queue, stock, hazards
    C->>API: POST /api/ops/wave
    API->>O: run_wave
    O->>AG: best_strategy, plan_round
    O->>O: simulator step with live stock and hazards
    O-->>C: deliveries, blocked, still waiting
    C->>API: POST /api/ops/sitrep
    API->>O: sitrep (Gemini, or rules)
    O-->>C: report text
```

**Operation state** (`ops.json`):

| Field | Contents |
|---|---|
| `incidents` | `id`, zone, coordinates, item, remaining `qty`, `requested`, urgency, `status` (`waiting` or `delivered`), source (`console` or `phone`), `located_by` (`gps` or `report`), `note`, report and delivery times |
| `stock` | Units per item per depot. Starts at 400 each and goes down with every delivery. |
| `hazards` | Hazards from reports, plus any bridge collapse from a wave. They stay until a new operation starts. |
| `deliveries`, `waves` | History: what each wave delivered, with which truck and strategy |
| `version` | Incremented on every change. Used to decide whether the browser copy is newer than the server's. |

**A wave** builds a simulator episode containing only the waiting incidents (no simulated requests, no random floods), the operation's stock and its hazards. It then plans one round with `agent.best_strategy`, the strategy with the best average score in memory (excluding `fifo`), and executes one simulator step.

Results are written back by incident id. The state is re-read under a lock, so reports that arrive while a wave is being planned are not lost:
- delivered quantities reduce `qty` and depot stock
- a fully served incident becomes `delivered`, with its truck, stop and arrival time in `note`
- blocked and no-truck incidents keep `waiting`, with the reason in `note`

**Storage and several instances** (`store.py`). With `FIREBASE_SERVICE_ACCOUNT` set, the operation and memory are Firestore documents (`reliefroute/ops`, `reliefroute/memory`, one string field `json`), read and written over the Firestore REST API with a service-account token. Every write carries the `updateTime` from its read as a precondition (`currentDocument.exists=false` for the first write). If another instance wrote in between, Firestore rejects the write and the read-modify-write is retried on fresh data, so a phone report and a wave planned on different instances never overwrite each other. Without Firestore the documents are local JSON files guarded by a process lock: fine for one instance, but two instances running at the same time each see their own state. The browser-copy restore is only used with file storage, and only adopts a copy from the same operation (or when the instance started empty), so a reset operation is never resurrected.

**GPS placement.** A phone report with a position inside the Chennai area places all of its incidents at that position. This replaces the coordinates Gemini inferred from the place name.

**Situation report.** `ops.sitrep` assembles the facts: figures, per-area waiting needs with wait times, hazards, stock and recent waves. Gemini turns them into headline, overview, priorities, blocked roads and actions, using only that data, and the output is validated and truncated field by field. Stock alerts are a threshold fact, so they always come from the rule and Gemini never sees raw stock levels; in testing it flagged healthy depots as alerts. Without Gemini, `_rule_sitrep` produces the same sections deterministically: priorities by urgency then wait time, stock under 100 units, and one action per open problem.
