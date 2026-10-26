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
    page["static/*.html<br/>console, reporter, driver, tracking"] -.->|fetch| main["main.py<br/>FastAPI routes, sign-in check"]
    ext["MCP clients"] -.->|JSON-RPC| main
    main --> auth["auth.py<br/>Firebase ID tokens"]
    main --> ops["ops.py<br/>live operations"]
    main --> auto["autopilot.py<br/>when to dispatch"]
    auto --> ops
    main --> mcp["mcp_server.py<br/>MCP protocol + tools"]
    main --> agent["agent.py<br/>decision logic"]
    mcp --> agent
    mcp --> ops
    ops --> agent
    ops --> roads["roads.py<br/>OSM road routing"]
    agent --> env["env.py<br/>flood simulator"]
    agent --> gemini["Gemini API<br/>via google-genai"]
    ops --> store["store.py"]
    agent --> store
    store --> fs[("Firestore<br/>or local JSON")]
```

| Module | Responsibility |
|---|---|
| `env.py` | Builds a flood scenario, executes delivery plans, injects the disruption, computes the reward. Pure Python, no I/O. |
| `agent.py` | Everything the agent decides: intake, strategy choice, planning, plan repair, model failover, travel times, reflection, memory. |
| `main.py` | HTTP layer. Turns requests into calls on `agent.py` and returns JSON. |
| `mcp_server.py` | Implements the MCP methods and maps five tools onto `agent.py` and `ops.py`. |
| `static/index.html` | The console: report form, pipeline, map, event log, performance panel. One file, no build step. |
| `policy.py` | Features and decision procedure of the learned policy, plus inference in plain Python from `policy_weights.json`. |
| `rl/train_policy.py` | Trains the policy with REINFORCE in PyTorch. Only needed offline. |
| `relief_openenv/` | The simulator wrapped as an OpenEnv environment (typed models, server, client, manifest). |
| `ops.py` | Live operations: incident queue and duplicate merging, dispatch waves (`plan_wave`, then `commit_wave`), routes, rerouting of trucks on the road, driver updates, resources, tracking, situation reports. |
| `autopilot.py` | Decides when a wave is worth sending; turns it into a proposal or sends it; approval, rejection, expiry and the planning lease. |
| `roads.py` | Road distance tables and truck routes from a Valhalla server on OpenStreetMap, with hazards as areas to avoid; straight-line fallback. |
| `auth.py` | Coordinator sign-in: verifies Firebase ID tokens and checks the coordinator list. |
| `store.py` | Shared state storage: Cloud Firestore (REST, conditional writes with retry) or local JSON files. |
| `static/report.html`, `static/track.html`, `static/driver.html` | Field reporter page, the reporter's status page, and the driver's route page. |

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
| Routes API fails (simulator ETAs) | HTTP error or timeout (8 s) | Estimate from distance at 25 km/h | `FALLBACK` |
| Road router unreachable or too many stops (live) | HTTP error, timeout (12 s), or more than 40 points | Plan and route on straight-line estimates; the route is drawn dashed and labelled "estimate" | `FALLBACK` |
| Driver reports a blocked road | Stop outcome `blocked` | New hazard at the driver's GPS position, incident back in the queue, next wave's router avoids the area | `BLOCKED`, `REROUTE` |
| Delivery fails or a route is recalled | Stop outcome `failed`, or recall | Incident back in the queue with the reason; unused cargo back into stock; truck free again | `DRIVER` |
| A truck leaves on another route while a wave is planned | Truck not idle when the wave is applied | That truck's stops stay queued for the next wave | `FALLBACK` |
| The same need is reported twice | Same item, open, same place (0.5 km between GPS fixes, same locality for guessed positions), within 12 hours | Merged into one request (larger quantity, higher urgency) | `MERGED` |
| Two instances write the operation at once | Firestore precondition fails | Re-read and retry the change on fresh data | – |
| Serverless instance lost its memory (file storage only) | Browser copy has more episodes than the server's | Server adopts the browser copy after sanitising it | – |

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
| `tools/list` | The five tools with JSON Schema inputs |
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

**Live hosting.** The app runs on Vercel as a serverless Python function (FastAPI zero-config). Instances start on demand and can be recycled, so state lives in Cloud Firestore:
- The operation and the agent's memory are two Firestore documents, shared by every instance (see section 12).
- Without Firestore (local development), they are `ops.json` and `memory.json`. On Vercel those are in `/tmp`, so an instance restart loses them. For memory, the browser's copy restores learning after such a restart.
- Training sends one episode per request, so no single request runs long and the UI can show progress.

**Cloud Run.** `Dockerfile` builds a container that serves the same app on `$PORT`. It is ready to deploy to Google Cloud Run once a billing-enabled project is available.

**Secrets.** The Gemini key and the Firestore service-account key are encrypted Vercel environment variables. Neither is in the repository. The Firebase web `apiKey` used for sign-in is public by design; it only identifies the project.

**Coordinator sign-in.** When `COORDINATOR_EMAILS` and `FIREBASE_WEB_API_KEY` are set:
- The console loads the Firebase Authentication SDK and signs coordinators in with Google.
- Every request carries the Firebase ID token. `auth.coordinator` verifies its signature, audience and expiry with Google's public keys, requires a verified email on the list, and caches the result until the token expires.
- Waves, resource changes, resets, situation reports and memory resets need a coordinator. Everyone else gets a read-only view without driver links.
- Reporters and drivers never sign in. A reporter's tracking link and a truck's driver link are 16-character random tokens. A new operation issues new driver links.

**Protecting a public deployment.** Rate limits keep one visitor from using up the shared Gemini quota or flooding the operation:

| Limit | Value | Where |
|---|---|---|
| Dispatches per client address | 8 per 10 minutes | `main.check_rate` |
| Training requests per client address | 60 per 10 minutes | `main.check_rate` |
| Costly MCP tool calls (`dispatch_relief`, `parse_field_report`, `train_agent`) | 10 per 10 minutes | `main.mcp` |
| Memory resets | 5 per 10 minutes | `main.check_rate` |
| Field reports | 12 per 10 minutes | `main.check_rate` |
| Waves | 30 per 10 minutes | `main.check_rate` |
| Situation reports | 6 per 10 minutes | `main.check_rate` |
| Resource changes (depots, trucks, restocks, recalls, hazards) | 60 per 10 minutes | `main.check_rate` |
| Driver stop updates | 120 per 10 minutes | `main.check_rate` |
| Gemini calls per server instance per UTC day | `GEMINI_DAILY_BUDGET`, default 300 | `agent._spend_budget` |

When a rate limit is hit, the response is HTTP 429 with `Retry-After`. MCP gets a JSON-RPC error `-32000`. When the daily budget is spent, every Gemini step falls back to its rule-based method, exactly as if the API were down. All counters are in memory per instance, so they are a guard, not an exact global quota.

## 11. Tests

`test_app.py` runs with `OFFLINE=1` and temporary state files. Gemini, the road router and Firebase sign-in are replaced by fakes where a test needs them, so no network access or key is needed.

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
| `test_operations_report_wave_driver_and_sitrep_over_the_api` | Over HTTP:<br/>- a phone report placed by GPS, with a working tracking link<br/>- a wave loading exactly the dispatched units off the depots and sending trucks out<br/>- the driver page and route, a driver closing a route<br/>- a repeated stop and an unknown link rejected<br/>- situation report, reset, empty-queue 409 |
| `test_duplicate_reports_merge_instead_of_sending_two_trucks` | Same need nearby merges (larger quantity, higher urgency, report count); a phone's GPS merges with a locality-level guess 1.2 km away; another place, or two GPS fixes 1 km apart, do not |
| `test_failed_and_blocked_stops_return_to_queue_and_teach_the_agent` | A short delivery, a blocked road and a failed delivery:<br/>- queue and notes<br/>- a hazard at the driver's GPS<br/>- cargo returned to stock, trucks free<br/>- lessons, the ETA correction learned from a late arrival, and the next wave using it |
| `test_recall_resources_and_their_guards` | Recall; new depot (outside Chennai refused); new truck (bad capacity refused); restock (unknown item refused); out of service (refused while on a route); closing a request |
| `test_hazards_are_avoided_by_the_router_and_can_be_cleared` | Against a fake router: hazards sent as excluded areas, road geometry decoded, a hazard cleared; with the router down, the wave falls back to estimates and logs it |
| `test_coordinator_sign_in_guards_every_change` | With sign-in on: no token, a wrong account and a forged token all get 401 on every protected action; reporters still report; driver links are hidden from the public view; the coordinator's email is recorded |
| `test_sitrep_stock_alerts_are_never_left_to_gemini` | Stock alerts come from the exact rule even when Gemini writes the report |
| `test_field_outcomes_choose_the_live_strategy` | One bad field route does not overturn the simulator; three routes per strategy do (84% vs 60% expected); the wave records the strategy, its reason and the scores; `/api/memory` reports the same choice |
| `test_intake_treats_the_report_as_data_and_caps_what_one_report_adds` | Gemini gets the instructions as a system instruction and the report as data it cannot close early; 10 needs and 5 hazards become 6 and 3, with a heal |
| `test_requests_without_a_known_place_are_flagged_and_never_merged` | No usable place: `located` is `guess`, the incident says so, and two such requests are not merged |
| `test_uploads_are_checked_for_type_and_size` | A text file as a photo gets 415; a file over 4 MB gets 413 |
| `test_blocked_stop_after_a_dropped_stop_number_does_not_crash` | A blocked report on stop 3 of a route without a stop 2 places the hazard halfway from the depot (an IndexError before) |
| `test_a_confirmed_hazard_reroutes_only_the_trucks_heading_into_it` | One report does not reroute; the second confirms the hazard and reroutes only the route that crosses it (new geometry, later ETA, `rev` 2, the log), not the far one |
| `test_a_driver_blocked_report_reroutes_other_trucks_over_the_api` | A driver's blocked road on one truck's way reroutes the other truck, over HTTP |
| `test_a_stop_cut_off_by_a_new_hazard_goes_back_to_the_queue_and_its_cargo_to_stock` | No road reaches the stop: it is `unreachable`, the request is waiting again with its full quantity, the route closes, every unit is back in stock, the strategy is not scored, the autopilot is told |
| `test_router_down_warns_the_driver_and_keeps_the_route` | The router fails: the route is unchanged, the driver gets a warning, a heal is logged |
| `test_autopilot_proposes_waits_for_approval_and_learns_from_rejection` | Off does nothing; auto is refused before an approval; propose makes one proposal and waits; approval sends exactly one wave; a rejection's reason becomes a lesson and starts the cool-down; an approval after 10 minutes is refused as expired |
| `test_autopilot_auto_mode_dispatches_and_skips_unlocated_requests` | Requests with no known place are never dispatched; a held lease blocks planning; auto mode sends a wave as the agent; a non-urgent request is batched until it has waited 30 minutes |
| `test_autopilot_endpoints_need_a_coordinator_or_the_scheduler_secret` | Every autopilot, proposal and confirm endpoint returns 401 without a coordinator; the scheduler secret runs a tick; the public view hides reporters' words, exact positions and coordinators' emails |

## 12. Live operations

`ops.py` turns the agent from "one report, one episode" into a running operation with real trucks, drivers and outcomes.

```mermaid
sequenceDiagram
    participant F as Reporter phone
    participant C as Console
    participant API as main.py
    participant O as ops.py
    participant AG as agent.py
    participant R as roads.py (OSM)
    participant D as Driver phone
    F->>API: POST /api/reports (text, photo, voice, GPS)
    API->>AG: intake
    API->>O: add_report: new incident, or merge into an open one
    O-->>F: tracking link /track/...
    C->>API: POST /api/ops/wave (coordinator), or the autopilot tick
    API->>O: plan_wave
    O->>R: distance table between depots and incidents, confirmed hazards avoided
    O->>AG: choose_live_strategy (field + simulator), plan_round (free trucks, real stock, road km)
    O->>R: one road route per truck
    O-->>C: a proposal to approve (autopilot), or straight to commit_wave
    O-->>C: routes and ETAs, stock loaded, trucks on route
    D->>API: GET /api/driver/... (route, stops, road geometry)
    D->>API: POST stop n: delivered / failed / blocked (+ GPS)
    API->>O: driver_update (queue, stock, hazard, truck position)
    API->>O: reroute_affected (a new hazard): new road routes for trucks heading into it
    O->>AG: record_field_route when the route closes
    C->>API: GET /api/ops every 5 s: driver events, reroutes and proposals appear in the log
```

**Operation state** (one document, `reliefroute/ops`):

| Field | Contents |
|---|---|
| `depots` | `id`, name, coordinates, `stock` per item. Starts with three depots at 400 units of each item; coordinators open more. |
| `trucks` | `id`, `depot_id`, `capacity`, `status` (`idle`, `on_route`, `out_of_service`), `route_id`, and `token` (the secret in its driver link) |
| `incidents` | `id`, zone, coordinates, item, `requested`, `qty` (still to send), `assigned` (on trucks), `delivered`, urgency, `status` (`waiting`, `en_route`, `delivered`, `closed`), `reports` (merged count), source, `located_by`, `note`, times |
| `routes` | `id`, wave, truck, depot, `status` (`active`, `done`), strategy, `km`, `minutes`, `source` (`road` or `estimate`), `shape` (road geometry, at most 300 points), `eta_factor`, `rev` (1, plus one per reroute), `reroutes` (time, hazards, km left before and after, stops cut off), `warning` (for the driver), `last_pos` (the truck's last GPS fix), and `stops` (`n`, incident, item, `qty`, `eta_raw`, `eta_min`, `eta_at`, `status` (`pending`, `delivered`, `failed`, `blocked`, `skipped`, `unreachable`), `delivered_qty`, `note`, `at`) |
| `hazards` | `id`, label, coordinates, `radius_km`, source (`console`, `phone`, `driver`), `active`, `verified` (false for a single report), `confirmations` |
| `autopilot` | `mode` (`off`, `propose`, `auto`), `min_urgency`, `max_wait_min`, `dirty` (a field change worth planning for), `lease_until`, `last_plan_at`, `considered` (incidents already planned for), `approved` (count) |
| `proposals` | The last 20: `id`, times, `expires_at`, `trigger`, `status` (`pending`, `approved`, `rejected`, `expired`, `withdrawn`), strategy and its reason, the planned routes (geometry dropped once decided), who decided and why |
| `reports` | One per field report: tracking `token`, time, source, the incidents it created or merged into |
| `waves`, `restocks`, `log` | History, incoming supplies, and an activity log of who did what |
| `version`, `schema` | Change counter; document format (an older format is replaced by a fresh operation on first use) |

An incident's status follows from its quantities: `waiting` while some of it still has to be sent (`qty > 0`), `en_route` while the rest is on trucks, `delivered` when both are zero, `closed` when a coordinator closed it.

**A wave**
1. It takes every waiting incident, the trucks that are `idle` (with their own capacities) and the depots' current stock.
2. It asks the road router for a distance table between all of those points, with every confirmed hazard as an area to avoid. While planning, `env.road_distances` installs that table, so the simulator, the rule strategies and the learned policy all see road kilometres. A pair the router connected is never treated as crossing a hazard, because its road route already goes around it.
3. It plans one round with the strategy from `agent.choose_live_strategy` inside a simulator episode that contains only these incidents, depots and trucks. For every strategy except `fifo`, the expected score is `(n_field × field_mean + 3 × sim_mean) / (n_field + 3)`, where field data counts once a strategy has two closed routes; without field data it is the simulator mean. The highest wins; there is no exploration in live operations. The wave stores the expected scores and a one-line reason.
4. For each truck with stops, it asks the router for the road route through the stops in order. That gives the geometry the driver sees and the drive time of every leg.
   - The ETA of stop *k* is the drive time up to it, plus 10 minutes of unloading at each earlier stop, multiplied by the learned `eta_factor`.
5. `commit_wave` applies the result in one atomic update, re-checking everything against fresh data (a report, a driver or another wave may have changed things meanwhile):
   - Stock is taken off the depot.
   - `qty` moves to `assigned`.
   - The truck becomes `on_route`.
   - A truck that became busy in between keeps its stops queued.

Steps 1–4 are `plan_wave`, which writes nothing; step 5 is `commit_wave`. A coordinator's wave runs both at once. The autopilot can run step 5 minutes later, when a coordinator approves its proposal; the re-check in step 5 is what makes that safe.

If the router fails or the wave has more than 40 points, planning and routes use straight-line estimates, and the log says which.

**Drivers** (`/driver/<token>`) see their route on a map, with the stops in order, ETAs and a link to open each stop in their navigation app. For each stop they record:

| Outcome | Effect |
|---|---|
| `delivered` (all, or a smaller quantity) | `delivered` grows; any shortfall goes back to `qty`, so it is still dispatched |
| `failed`, with a reason | The quantity goes back to `qty`; the reason is shown on the incident |
| `blocked`, with a note and GPS | As failed, plus a new confirmed 0.5 km hazard at the driver's position (or halfway from the last stop the truck reached, or its depot, without GPS), and rerouting of every truck heading into it |

Every update carries the phone's GPS position when the driver allows it; it becomes the route's `last_pos`.

When the last stop is recorded the route closes:
- Unused cargo returns to the depot's stock.
- The truck is free again.
- The outcome goes to `agent.record_field_route`.

A coordinator can also recall a route; its unvisited stops are `skipped` and return to the queue.

**Rerouting trucks on the road** (`ops.reroute_affected`). It runs when a hazard becomes one that routes avoid: a driver reports a blocked road, a second report confirms a reported hazard, or a coordinator confirms one. For each active route with stops left:
1. **Where is the truck?** Its `last_pos`, else the last stop it reached, else its depot. A point inside an avoided area is skipped: the router cannot start there, and the truck has to turn back out of it.
2. **Does the rest of the route cross the new hazard?** For a road route, any point of its geometry from the point nearest the truck onwards within the hazard's radius plus 0.1 km. For an estimated route, any straight leg from the truck through the remaining stops. A hazard that covers a stop itself cannot be routed around; the driver gets a warning instead.
3. **New route.** The router is asked for a road route from the truck through the remaining stops, around every confirmed hazard. If it answers that no path exists, each stop is tried on its own: stops it cannot reach are marked `unreachable`, their quantity goes back to the incident's `qty`, and their cargo stays on the truck until the route closes. If no stop is left, the route closes.
4. **Commit.** One atomic update, only if the route is still active, its `rev` unchanged and its pending stops the same. Pending stops get new leg distances, `eta_raw` and `eta_at` (now plus the new drive time, times `eta_factor`); the route gets the new geometry and `rev + 1`, and the change is logged. If the router could not answer, the route is kept and `warning` tells the driver to drive around the hazard.

At most 5 routes are rerouted per hazard. The driver's page compares `rev` on every refresh: when it grows, it vibrates, shows **Route changed** with the extra distance and any stops cut off, and redraws the map. Stops cut off this way are not held against the strategy.

**Autopilot** (`autopilot.py`). `tick()` first reads the operation and decides, without writing, whether a wave is due (`should_plan`):
- never when the mode is `off`, a proposal is pending, another instance holds the lease, or the last plan was under 2 minutes ago
- never without a waiting request with a known place, or without a free truck
- yes when `dirty` is set (a truck came back, a stop failed or was blocked, a stop was cut off), when there is an urgent request (urgency ≥ `min_urgency`, default 4) it has not planned for yet, or when a request has waited `max_wait_min` (default 30) and the last plan is that old. Other requests are batched.

When a wave is due, it claims a 90-second lease with a conditional write (the same mechanism that protects every update, so two instances cannot both claim it), runs `plan_wave` without requests whose place is unknown, then:
- in `propose` mode stores a proposal that expires after 10 minutes. Approving runs `commit_wave` with the stored plan; the proposal is marked approved inside the same atomic write, so a double click or two coordinators cannot send two waves. Rejecting stores the reason, restarts the cool-down, and adds the reason to the lessons the Gemini planner reads.
- in `auto` mode commits the wave as `agent (autopilot)`. Auto mode can only be switched on after one approved proposal.

Ticks come from the coordinator console every 15 seconds while it is open, or from any scheduler that calls `POST /api/agent/tick` with `X-Agent-Secret`. Serverless instances do not run background work, so nothing is planned while neither is calling.

**Intake safety.** Gemini receives the intake instructions as a system instruction and the report wrapped in `<report>` tags (any closing tag inside the report is removed), and is told the report is data to extract from. One report adds at most 6 needs and 3 hazards. A hazard from a report starts unconfirmed (`verified: false`): it is shown on maps and to drivers, but routes avoid it only after a second report within 1 km, a driver's report or a coordinator's confirmation. A need whose place could not be read is placed at the city centre with `located_by: guess`; it is not drawn on the map, never merged with another, and never dispatched by the autopilot. Uploads must be image or audio files of at most 4 MB; every Gemini call has a 25-second timeout.

**The public view.** `ops.view` serves non-coordinators the operation without driver links, without reporters' summaries and evidence, with incident, stop and route positions rounded to 3 decimals (about 100 m), without truck positions, and with coordinators' email addresses replaced by "coordinator".

**Learning from the field** (`agent.record_field_route`):
- **Score:** the urgency-weighted share of the loaded relief that was delivered, averaged per strategy in `memory.field`. Stops `blocked` or `unreachable` because of a road cut after dispatch are left out; a route with nothing else is not scored. This score chooses later waves' strategy (see step 3 of a wave).
- **ETA correction:** each delivered stop with a plausible arrival time moves `eta_factor` 20% of the way towards actual / router estimate, clamped to 0.5–3. A stop counts when it was recorded at least 30% of the estimate after dispatch, so taps seconds after dispatch do not count. Every later ETA uses the new factor.
- **Lessons:** failed and blocked stops become lessons ("road to Velachery was blocked …"), which the Gemini planner reads with the others.

**Duplicate merging.** A new need is merged into an existing incident when all of these hold:
- same item
- the incident is still open
- the same place: within 0.5 km when both positions are GPS. A position guessed from a place name is only precise to the locality, so then the same locality name, or within 1.5 km, is enough. (Live testing showed Gemini's guess for "Velachery" 1.2 km from a phone's GPS in Velachery.)
- reported in the last 12 hours

The larger requested quantity and the higher urgency win, a GPS position improves a guessed one, and `reports` counts the reports. The reporter is told their report was added to an existing request, and their tracking link follows it.

**Tracking** (`/track/<token>`) shows a reporter only the incidents from their own report: received, truck on the way (with the earliest ETA), delivered, and whether an earlier truck could not get through.

**Storage and several instances** (`store.py`). With `FIREBASE_SERVICE_ACCOUNT` set, the operation and memory are Firestore documents (`reliefroute/ops`, `reliefroute/memory`, one string field `json`), read and written over the Firestore REST API with a service-account token.
- Every write carries the `updateTime` from its read as a precondition (`currentDocument.exists=false` for the first write).
- If another instance wrote in between, Firestore rejects the write and the change is retried on fresh data. So a phone report, a wave and a driver update handled by three different instances never overwrite each other.
- An update that fails validation (for example a stop already recorded) raises before anything is written.
- Without Firestore the documents are local JSON files guarded by a process lock. That is fine for one instance, not for several.

**Situation report.** `ops.sitrep` assembles the facts: figures, per-area waiting needs with wait times, relief on trucks, active routes, hazards, stock and recent waves. Gemini turns them into headline, overview, priorities, blocked roads and actions, using only that data, and the output is validated and truncated field by field. Stock alerts are a threshold fact, so they always come from the rule and Gemini never sees raw stock levels; in testing it flagged healthy depots as alerts. Without Gemini, `_rule_sitrep` produces the same sections deterministically, including routes out for more than two hours.
