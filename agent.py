"""ReliefRoute agent: multimodal intake -> plan -> dispatch -> self-heal -> reflect/learn.

Learning happens at two levels, both persisted in memory.json:
  1. Policy selection: an epsilon-greedy bandit (tiny RL) over planning strategies,
     updated with each episode's reward from the simulator.
  2. Reflection: after each episode Gemini writes lessons that are fed into the
     next episode's planning prompt.
"""
import json
import os
import random
import re
import threading
import time

import httpx

from env import CENTER, ITEMS, ZONES, ReliefEnv, distance_km

STRATEGIES = ["llm", "urgency_first", "nearest_first", "fifo"]
EPSILON = 0.2
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_lock = threading.Lock()


def _gemini_key():
    if os.getenv("OFFLINE") == "1":
        return None
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _maps_key():
    if os.getenv("OFFLINE") == "1":
        return None
    return os.getenv("MAPS_API_KEY") or _gemini_key()


def _models():
    return [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-flash-latest,gemini-3.8-flash,gemini-3.5-flash-lite").split(",") if m.strip()]


def llm_available():
    return bool(_gemini_key())


# ---- memory -----------------------------------------------------------------
def _memory_path():
    # Serverless hosts (Vercel) only allow writes under /tmp
    return os.getenv("MEMORY_PATH") or ("/tmp/memory.json" if os.getenv("VERCEL") else "memory.json")


def load_memory():
    try:
        with open(_memory_path(), encoding="utf-8") as f:
            mem = json.load(f)
    except (OSError, ValueError):
        mem = {}
    mem.setdefault("lessons", [])
    mem.setdefault("episodes", [])
    mem.setdefault("bandit", {})
    for s in STRATEGIES:
        mem["bandit"].setdefault(s, {"n": 0, "mean": 0.0})
    return mem


def save_memory(mem):
    with open(_memory_path(), "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=1)


def sync_memory(client_mem):
    """Adopt the browser's copy of memory when it is ahead of ours.

    Serverless instances lose /tmp when they restart; the browser keeps a copy so
    learning survives cold starts.
    """
    if not isinstance(client_mem, dict) or not isinstance(client_mem.get("episodes"), list):
        return
    with _lock:
        mem = load_memory()
        if len(client_mem["episodes"]) <= len(mem["episodes"]):
            return
        bandit = {s: {"n": int(v.get("n", 0)), "mean": float(v.get("mean", 0.0))}
                  for s, v in (client_mem.get("bandit") or {}).items()
                  if s in STRATEGIES and isinstance(v, dict)}
        save_memory({"lessons": [str(l)[:160] for l in client_mem.get("lessons") or []][-8:],
                     "episodes": [e for e in client_mem["episodes"] if isinstance(e, dict)][-200:],
                     "bandit": bandit})


def reset_memory():
    with _lock:
        save_memory(load_memory() | {"lessons": [], "episodes": [], "bandit": {}})


def select_strategy(mem, explore=True):
    arms = [s for s in STRATEGIES if s != "llm" or llm_available()]
    untried = [s for s in arms if mem["bandit"][s]["n"] == 0]
    if untried:
        return untried[0]
    if explore and random.random() < EPSILON:
        return random.choice(arms)
    return max(arms, key=lambda s: mem["bandit"][s]["mean"])


# ---- Gemini (with self-healing retries + model fallback) -------------------
def _parse_json(text):
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


QUOTA_COOLDOWN_S = 300
_cooldown = {}  # model -> time until which it is skipped (circuit breaker for exhausted quota)


def gemini_json(parts, heals):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_gemini_key())
    last = None
    now = time.time()
    models = [m for m in _models() if _cooldown.get(m, 0) <= now] or _models()  # all cooling down: try anyway
    for model in models:
        for attempt in range(2):
            try:
                r = client.models.generate_content(
                    model=model, contents=parts,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2))
                return _parse_json(r.text)
            except Exception as e:  # network, quota, bad JSON -> heal by retry/fallback
                last = e
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    # retrying a spent quota is pointless: trip the breaker and route to the next model
                    _cooldown[model] = time.time() + QUOTA_COOLDOWN_S
                    heals.append(f"Gemini {model} quota exhausted -> circuit breaker routes to next model "
                                 f"for {QUOTA_COOLDOWN_S // 60} min")
                    break
                heals.append(f"Gemini {model} attempt {attempt + 1} failed ({str(e)[:90]}) -> retrying")
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"all Gemini models failed: {last}")


# ---- multimodal intake ------------------------------------------------------
INTAKE_PROMPT = """You are the intake officer of a Chennai flood-relief logistics operation.
From the attached field report (any mix of text, photo, voice note; any language incl. Tamil/Hindi),
extract every relief request and every road hazard you can see or hear.
Return ONLY JSON:
{"summary": "<one line in English>", "language": "<detected>",
 "needs": [{"zone": "<Chennai locality>", "lat": <float>, "lng": <float>,
            "item": "water|food|medicine|blankets", "qty": <int people/units>, "urgency": <1-5>,
            "evidence": "<what in the input told you this>"}],
 "hazards": [{"label": "<e.g. road flooded>", "lat": <float>, "lng": <float>}]}
Use approximate real coordinates for the named Chennai locality. Urgency 5 = life-threatening.
Text from the field: """


def _zone_coords(name):
    for z, la, lo in ZONES:
        if name and z.lower() in name.lower():
            return la, lo
    return None


def _clean_need(n):
    item = str(n.get("item", "food")).lower()
    item = next((i for i in ITEMS if i in item), "food")
    lat, lng = n.get("lat"), n.get("lng")
    if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))
            and 12.8 < lat < 13.35 and 80.0 < lng < 80.4):
        lat, lng = _zone_coords(n.get("zone")) or CENTER
    try:
        qty = max(1, min(1000, int(n.get("qty", 50))))
    except (TypeError, ValueError):
        qty = 50
    try:
        urg = max(1, min(5, int(n.get("urgency", 3))))
    except (TypeError, ValueError):
        urg = 3
    return {"zone": str(n.get("zone") or "Unknown"), "lat": lat, "lng": lng, "item": item, "qty": qty,
            "urgency": urg, "evidence": str(n.get("evidence", ""))[:200]}


ITEM_WORDS = {  # whole words only, so "underwater" is not a water request
    "water": ["water", "drinking", "thanni", "kudineer", "paani", "pani"],
    "food": ["food", "meals", "rice", "packets", "saapadu", "khana"],
    "medicine": ["medicine", "medicines", "medical", "tablets", "first aid", "marundhu", "dawai"],
    "blankets": ["blanket", "blankets", "bedsheets", "kambal", "porvai"],
}


def _offline_intake(text):
    t = text.lower()
    found = []
    for item, words in ITEM_WORDS.items():
        hits = [m.start() for w in words for m in re.finditer(rf"\b{re.escape(w)}\b", t)]
        if hits:
            found.append((min(hits), item))
    items = [i for _, i in sorted(found)] or ["food"]
    m = re.search(r"\d+", t)
    zone = next((z for z, _, _ in ZONES if z.lower() in t), "T Nagar")
    urgent = any(w in t for w in ("urgent", "trapped", "injured", "emergency", "children", "stranded"))
    needs = [{"zone": zone, "item": item, "qty": int(m.group(0)) if m else 50, "urgency": 5 if urgent else 3,
              "evidence": "keyword rules (AI unavailable)"} for item in items] if t.strip() else []
    return {"summary": text[:120] or "No text report", "language": "unknown", "needs": needs, "hazards": []}


def intake(text="", image=None, audio=None, heals=None):
    """image/audio: (bytes, mime_type) or None."""
    heals = heals if heals is not None else []
    if not llm_available():
        out = _offline_intake(text)
    else:
        from google.genai import types
        parts = [INTAKE_PROMPT + (text or "(none)")]
        for media in (image, audio):
            if media and media[0]:
                parts.append(types.Part.from_bytes(data=media[0], mime_type=media[1]))
        try:
            out = gemini_json(parts, heals)
        except Exception as e:
            heals.append(f"Multimodal intake failed ({str(e)[:80]}) -> keyword fallback")
            out = _offline_intake(text)
    needs = [_clean_need(n) for n in out.get("needs", []) if isinstance(n, dict)]
    hazards = [h for h in out.get("hazards", []) if isinstance(h, dict)
               and isinstance(h.get("lat"), (int, float)) and isinstance(h.get("lng"), (int, float))]
    return {"summary": out.get("summary", ""), "language": out.get("language", ""), "needs": needs,
            "hazards": hazards}


# ---- planning ---------------------------------------------------------------
STRATEGY_NOTES = {
    "urgency_first": "Most urgent first; nearest depot with enough stock; detour around reported floods.",
    "nearest_first": "Arrival order; nearest depot with enough stock; detour around reported floods.",
    "fifo": "Arrival order; first depot with any stock; ignores flood reports.",
}


def heuristic_plan(obs, strategy):
    stock = {d["id"]: dict(d["stock"]) for d in obs["depots"]}
    needs = obs["pending_needs"]
    if strategy == "urgency_first":
        needs = sorted(needs, key=lambda n: -n["urgency"])
    out = []
    for n in needs:
        opts = obs["route_matrix"][n["id"]]
        ids = [d["id"] for d in obs["depots"] if stock[d["id"]].get(n["item"], 0) > 0]
        if not ids:
            continue
        if strategy == "fifo":  # naive: first depot with stock, ignores hazards
            did, detour = ids[0], False
        else:
            did = min(ids, key=lambda i: (stock[i][n["item"]] < n["qty"],
                                          opts[i]["km"] * (1.6 if opts[i]["blocked_by"] else 1)))
            detour = bool(opts[did]["blocked_by"])
        stock[did][n["item"]] -= min(n["qty"], stock[did][n["item"]])
        out.append({"need_id": n["id"], "depot_id": did, "detour": detour})
    return out


PLAN_PROMPT = """You are the dispatch planner of a flood-relief operation. Assign each pending need to one depot.
Goals: serve the most urgent needs first, never exceed depot stock, prefer short routes, and set
"detour": true whenever route_matrix shows the route is blocked_by a hazard (detours cost 60% more distance).
Lessons learned from previous episodes (apply them!):
{lessons}
Current observation:
{obs}
Return ONLY JSON: {{"reasoning": "<2 sentences>", "assignments": [{{"need_id": "...", "depot_id": "...", "detour": false}}]}}"""


def llm_plan(obs, lessons, heals):
    compact = {k: obs[k] for k in ("step", "depots", "known_hazards", "route_matrix")}
    compact["pending_needs"] = [{k: n[k] for k in ("id", "zone", "item", "qty", "urgency")} for n in obs["pending_needs"]]
    prompt = PLAN_PROMPT.format(lessons="\n".join(f"- {l}" for l in lessons) or "- (none yet)",
                                obs=json.dumps(compact))
    data = gemini_json([prompt], heals)
    return repair(obs, data.get("assignments") or [], heals), str(data.get("reasoning", ""))


def repair(obs, assignments, heals):
    """Guardrail: validate an LLM plan against the observation and heal what is wrong."""
    pend = {n["id"] for n in obs["pending_needs"]}
    depots = {d["id"] for d in obs["depots"]}
    good, seen = [], set()
    for a in assignments if isinstance(assignments, list) else []:
        if not (isinstance(a, dict) and a.get("need_id") in pend and a.get("depot_id") in depots
                and a["need_id"] not in seen):
            heals.append(f"Guardrail dropped invalid assignment {json.dumps(a)[:80]}")
            continue
        blocked = obs["route_matrix"][a["need_id"]][a["depot_id"]]["blocked_by"]
        detour = bool(a.get("detour"))
        if blocked and not detour:
            heals.append(f"Guardrail: {a['need_id']} via {a['depot_id']} crosses {blocked} -> forcing detour")
            detour = True
        seen.add(a["need_id"])
        good.append({"need_id": a["need_id"], "depot_id": a["depot_id"], "detour": detour})
    missing = pend - seen
    if missing:
        heals.append(f"Planner skipped {sorted(missing)} -> filled by urgency_first fallback")
        sub = {**obs, "pending_needs": [n for n in obs["pending_needs"] if n["id"] in missing]}
        good += heuristic_plan(sub, "urgency_first")
    return good


# ---- Google Maps Routes (with fallback) ------------------------------------
def maps_eta(origin, dest, heals):
    key = _maps_key()
    if key:
        try:
            ll = lambda p: {"location": {"latLng": {"latitude": p[0], "longitude": p[1]}}}
            r = httpx.post(ROUTES_URL, timeout=8, json={"origin": ll(origin), "destination": ll(dest),
                                                         "travelMode": "DRIVE"},
                           headers={"X-Goog-Api-Key": key,
                                    "X-Goog-FieldMask": "routes.duration,routes.distanceMeters"})
            r.raise_for_status()
            route = r.json()["routes"][0]
            return {"km": round(route["distanceMeters"] / 1000, 1),
                    "minutes": round(int(route["duration"].rstrip("s")) / 60), "source": "google_routes"}
        except Exception as e:
            heals.append(f"Routes API failed ({str(e)[:70]}) -> distance estimate")
    km = distance_km(origin, dest)
    return {"km": round(km, 1), "minutes": round(km / 25 * 60), "source": "estimate"}


# ---- reflection ---------------------------------------------------------------
def reflect(trace, total, strategy, heals, lessons):
    blocked = sum(len(t["info"]["blocked"]) for t in trace)
    unmet = trace[-1]["info"].get("unmet", [])
    if strategy == "llm" and llm_available():  # heuristic episodes reflect by rules: keeps training fast
        summary =[{"step": t["step"], "reward": t["reward"], "blocked": t["info"]["blocked"],
                    "stockout": t["info"]["stockout"], "unmet": t["info"].get("unmet")} for t in trace]
        prompt = ("You are improving a flood-relief dispatch agent. Episode result: strategy=" + strategy +
                  f", total_reward={total}, self-heal events={heals[:8]}, steps={json.dumps(summary)}.\n"
                  f"Existing lessons: {lessons}\nWrite 1-2 NEW short, concrete, reusable lessons (max 20 words each) "
                  'that would raise the reward next time. Return ONLY JSON {"lessons": ["..."]}')
        try:
            return [str(l)[:160] for l in gemini_json([prompt], heals).get("lessons", [])][:2]
        except Exception:
            heals.append("Reflection via Gemini failed -> rule-based reflection")
    out = []
    if blocked:
        out.append(f"{strategy}: {blocked} deliveries hit hazards; check blocked_by and detour before dispatch.")
    if unmet:
        out.append(f"{len(unmet)} needs left unmet; serve urgency 4-5 first and split across depots.")
    return out or [f"{strategy} scored {total}; keep shortest unblocked depot per need."]


# ---- full episode -------------------------------------------------------------
def run_episode(seed=None, extra_needs=None, extra_hazards=None, chaos=True, strategy=None,
                use_maps=False, explore=True):
    env = ReliefEnv()
    obs = env.reset(seed=seed, extra_needs=extra_needs, extra_hazards=extra_hazards, chaos=chaos)
    with _lock:
        mem = load_memory()
    strategy = strategy or select_strategy(mem, explore)
    if strategy == "llm" and not llm_available():
        strategy = "urgency_first"
    initial = {"depots": obs["depots"], "needs": [dict(n) for n in obs["pending_needs"]]}
    trace, heals = [], []
    while True:
        try:
            if strategy == "llm":
                action, why = llm_plan(obs, mem["lessons"], heals)
            else:
                action, why = heuristic_plan(obs, strategy), STRATEGY_NOTES[strategy]
        except Exception as e:
            heals.append(f"Planner crashed ({str(e)[:80]}) -> urgency_first fallback")
            action, why = heuristic_plan(obs, "urgency_first"), "Fallback: " + STRATEGY_NOTES["urgency_first"]
        res = env.step({"assignments": action})
        info = res["info"]
        for h in info["new_hazards"]:
            heals.append(f"ALERT step {env.state()['step']}: {h['label']} detected mid-route")
        for b in info["blocked"]:
            heals.append(f"Delivery {b['need_id']} from {b['depot_id']} blocked by {b['hazards']} -> replanning")
        trace.append({"step": env.state()["step"], "strategy": strategy, "reasoning": why,
                      "action": action, "reward": res["reward"], "info": info})
        obs = res["observation"]
        if res["done"]:
            break
    st = env.state()
    total = st["total_reward"]
    # % of the best possible outcome, so episodes of different difficulty are comparable
    score = round(max(0.0, 100 * total / st["max_reward"]), 1) if st["max_reward"] else 0.0
    new_lessons = reflect(trace, total, strategy, heals, mem["lessons"])

    deliveries = st["deliveries"]
    if use_maps:
        needs = {n["id"]: n for n in st["needs"]}
        depots = {d["id"]: d for d in st["depots"]}
        for d in deliveries:
            if needs[d["need_id"]]["source"] == "field":
                n, dp = needs[d["need_id"]], depots[d["depot_id"]]
                d["eta"] = maps_eta((dp["lat"], dp["lng"]), (n["lat"], n["lng"]), heals)

    with _lock:
        mem = load_memory()
        arm = mem["bandit"][strategy]
        arm["n"] += 1
        arm["mean"] = round(arm["mean"] + (score - arm["mean"]) / arm["n"], 2)
        for l in new_lessons:
            if l not in mem["lessons"]:
                mem["lessons"].append(l)
        mem["lessons"] = mem["lessons"][-8:]
        n = mem["episodes"][-1]["n"] + 1 if mem["episodes"] else 1
        mem["episodes"].append({"n": n, "strategy": strategy, "reward": total, "score": score,
                                "heals": len(heals), "ts": int(time.time())})
        mem["episodes"] = mem["episodes"][-200:]
        save_memory(mem)

    return {"strategy": strategy, "total_reward": total, "score": score, "trace": trace, "heals": heals,
            "new_lessons": new_lessons, "initial": initial, "hazards": st["hazards"],
            "deliveries": deliveries, "unmet": trace[-1]["info"].get("unmet", []),
            "llm": llm_available()}


def train(episodes=5, chaos=True):
    results = [run_episode(chaos=chaos) for _ in range(episodes)]
    summary = [{"strategy": r["strategy"], "reward": r["total_reward"], "score": r["score"],
                "heals": len(r["heals"])} for r in results]
    return summary, results[-1]


def preview(seed=7):
    """Initial scenario for the map before any episode has run."""
    env = ReliefEnv()
    obs = env.reset(seed=seed, chaos=False)
    return {"initial": {"depots": obs["depots"], "needs": obs["pending_needs"]}, "hazards": obs["known_hazards"],
            "trace": [], "deliveries": [], "unmet": []}
