"""Live operations: field incidents are queued and dispatched in waves; coordinators get situation reports.

State is one JSON document (ops.json, /tmp on Vercel). The coordinator's browser keeps a copy and
sends it back, so a serverless restart can be recovered (same pattern as agent memory).
"""
import json
import os
import threading
import time

import agent
from env import ITEMS, ReliefEnv

OPS_STOCK = 400  # units of each item per depot when an operation starts
LOW_STOCK = 100
CHENNAI = (12.8, 13.35, 80.0, 80.4)  # lat min, lat max, lng min, lng max
INCIDENT_KEYS = {"id": str, "zone": str, "lat": float, "lng": float, "item": str, "qty": int, "requested": int,
                 "urgency": int, "status": str, "source": str, "summary": str, "evidence": str, "note": str,
                 "located_by": str, "reported_at": int, "delivered_at": int}
_lock = threading.Lock()


def _path():
    return os.getenv("OPS_PATH") or ("/tmp/ops.json" if os.getenv("VERCEL") else "ops.json")


def _depots():
    obs = ReliefEnv().reset(seed=0, n_needs=0, chaos=False, random_hazards=False)
    return obs["depots"]


def new_state():
    return {"version": 0, "next_id": 1, "started": int(time.time()), "incidents": [],
            "stock": {d["id"]: {it: OPS_STOCK for it in ITEMS} for d in _depots()},
            "hazards": [], "deliveries": [], "waves": []}


def load():
    try:
        with open(_path(), encoding="utf-8") as f:
            st = json.load(f)
        if isinstance(st, dict) and "incidents" in st:
            return st
    except (OSError, ValueError):
        pass
    return new_state()


def save(st):
    with open(_path(), "w", encoding="utf-8") as f:
        json.dump(st, f)


def reset():
    with _lock:
        save(new_state())
    return load()


def _in_chennai(lat, lng):
    return CHENNAI[0] < lat < CHENNAI[1] and CHENNAI[2] < lng < CHENNAI[3]


def sync(client):
    """Adopt the browser's copy when it is ahead of ours (serverless instances lose /tmp on restart)."""
    if not isinstance(client, dict) or not isinstance(client.get("incidents"), list):
        return
    try:
        version = int(client.get("version", 0))
    except (TypeError, ValueError):
        return
    with _lock:
        st = load()
        if version <= st["version"]:
            return
        clean = new_state()
        for inc in client["incidents"][-500:]:
            if not isinstance(inc, dict):
                continue
            try:
                row = {k: (t(inc[k]) if inc.get(k) is not None else None) for k, t in INCIDENT_KEYS.items()}
            except (TypeError, ValueError, KeyError):
                continue
            if row["item"] in ITEMS and row["status"] in ("waiting", "delivered") and _in_chennai(row["lat"], row["lng"]):
                row["urgency"] = max(1, min(5, row["urgency"]))
                clean["incidents"].append(row)
        for d, items in (client.get("stock") or {}).items():
            if d in clean["stock"] and isinstance(items, dict):
                clean["stock"][d] = {it: max(0, min(10_000, int(items.get(it, 0)))) for it in ITEMS}
        clean["hazards"] = [h for h in (client.get("hazards") or [])[-100:] if isinstance(h, dict)
                            and isinstance(h.get("lat"), (int, float)) and isinstance(h.get("lng"), (int, float))]
        clean["deliveries"] = [d for d in (client.get("deliveries") or [])[-500:] if isinstance(d, dict)]
        clean["waves"] = [w for w in (client.get("waves") or [])[-100:] if isinstance(w, dict)]
        clean["version"], clean["started"] = version, int(client.get("started") or clean["started"])
        clean["next_id"] = max([int(i["id"][1:]) for i in clean["incidents"] if i["id"][1:].isdigit()] + [0]) + 1
        save(clean)


def add_report(report, source="console", coords=None):
    """Turn an intake result into waiting incidents. A phone's GPS position overrides guessed coordinates."""
    gps = coords if coords and _in_chennai(*coords) else None
    now = int(time.time())
    with _lock:
        st = load()
        made = []
        for n in report["needs"]:
            inc = {"id": f"I{st['next_id']}", "zone": n["zone"], "lat": gps[0] if gps else n["lat"],
                   "lng": gps[1] if gps else n["lng"], "item": n["item"], "qty": n["qty"], "requested": n["qty"],
                   "urgency": n["urgency"], "status": "waiting", "source": source,
                   "summary": str(report.get("summary", ""))[:200], "evidence": str(n.get("evidence", ""))[:200],
                   "note": "", "located_by": "gps" if gps else "report", "reported_at": now, "delivered_at": None}
            st["next_id"] += 1
            st["incidents"].append(inc)
            made.append(inc)
        for h in report.get("hazards", []):
            st["hazards"].append({"label": str(h.get("label", "Reported hazard"))[:80], "lat": h["lat"],
                                  "lng": h["lng"], "radius_km": 0.8, "source": source, "reported_at": now})
        st["version"] += 1
        save(st)
    return made, st


def run_wave(chaos=False):
    """Plan and dispatch one round over every waiting incident with the best learned strategy."""
    heals = []
    with _lock:
        st = load()
    waiting = [i for i in st["incidents"] if i["status"] == "waiting" and i["qty"] > 0]
    if not waiting:
        return {"error": "No waiting incidents to dispatch.", "ops": st}
    mem = agent.load_memory()
    strategy = agent.best_strategy(mem)
    env = ReliefEnv()
    env.reset(n_needs=0, chaos=chaos, random_hazards=False, extra_hazards=st["hazards"],
              extra_needs=[{k: i[k] for k in ("id", "zone", "lat", "lng", "item", "qty", "requested", "urgency")}
                           for i in waiting])
    for d in env._s["depots"]:
        d["stock"] = dict(st["stock"][d["id"]])
    obs = env.observation()
    initial = {"depots": obs["depots"], "needs": [dict(n) for n in obs["pending_needs"]]}
    action, why = agent.plan_round(obs, strategy, mem["lessons"], heals)
    res = env.step({"assignments": action})
    info = res["info"]
    for h in info["new_hazards"]:
        heals.append(f"ALERT wave: {h['label']} detected mid-route")
    for b in info["blocked"]:
        heals.append(f"Delivery {b['need_id']} from {b['depot_id']} blocked by {b['hazards']} -> stays queued for the next wave")

    now = int(time.time())
    with _lock:  # re-read: reports may have arrived while the wave was being planned
        st = load()
        wave_no = len(st["waves"]) + 1
        byid = {i["id"]: i for i in st["incidents"]}
        for d in info["delivered"]:
            st["stock"][d["depot_id"]][d["item"]] = max(0, st["stock"][d["depot_id"]][d["item"]] - d["qty"])
            st["deliveries"].append({**d, "wave": wave_no, "dispatched_at": now})
            inc = byid.get(d["need_id"])
            if inc:
                inc["qty"] = max(0, inc["qty"] - d["qty"])
                if inc["qty"] == 0:
                    inc.update(status="delivered", delivered_at=now + d["minutes"] * 60,
                               note=f"wave {wave_no}: {d['truck_id']} stop {d['stop']}, arrives in {d['minutes']} min")
                else:
                    inc["note"] = f"wave {wave_no}: {d['qty']} delivered, {inc['qty']} still needed"
        for b in info["blocked"]:
            if b["need_id"] in byid:
                byid[b["need_id"]]["note"] = f"wave {wave_no}: route blocked by {', '.join(b['hazards'])}; retry next wave"
        for b in info["no_truck"] + info["stockout"]:
            if b["need_id"] in byid and byid[b["need_id"]]["status"] == "waiting":
                byid[b["need_id"]]["note"] = f"wave {wave_no}: no truck space or stock at {b['depot_id']}; retry next wave"
        for h in info["new_hazards"]:  # a collapsed bridge stays collapsed for later waves
            st["hazards"].append({"label": h["label"], "lat": h["lat"], "lng": h["lng"], "radius_km": h["radius_km"],
                                  "source": "operations", "reported_at": now})
        wave = {"n": wave_no, "at": now, "strategy": strategy, "reasoning": why, "open_before": len(waiting),
                "delivered": len(info["delivered"]), "units": sum(d["qty"] for d in info["delivered"]),
                "blocked": len(info["blocked"]), "no_truck": len(info["no_truck"])}
        st["waves"].append(wave)
        st["version"] += 1
        save(st)
    still = [i["id"] for i in st["incidents"] if i["status"] == "waiting"]
    return {"wave": wave, "strategy": strategy, "initial": initial, "hazards": env.state()["hazards"],
            "trace": [{"step": 1, "strategy": strategy, "reasoning": why, "action": action,
                       "reward": res["reward"], "info": info}],
            "deliveries": info["delivered"], "unmet": still, "heals": heals, "ops": st}


# ---- situation report --------------------------------------------------------------
def _ist(ts):
    return time.strftime("%d %b %Y %H:%M IST", time.gmtime(ts + 19800))


def summary(st):
    now = int(time.time())
    waiting = [i for i in st["incidents"] if i["status"] == "waiting"]
    return {"waiting": len(waiting), "delivered": sum(i["status"] == "delivered" for i in st["incidents"]),
            "units_delivered": sum(d.get("qty", 0) for d in st["deliveries"]), "hazards": len(st["hazards"]),
            "waves": len(st["waves"]),
            "longest_wait_min": max([(now - i["reported_at"]) // 60 for i in waiting] or [0])}


def _sitrep_data(st):
    now = int(time.time())
    names = {d["id"]: d["name"] for d in _depots()}
    areas = {}
    for i in st["incidents"]:
        a = areas.setdefault(i["zone"], {"zone": i["zone"], "waiting": [], "delivered_units": 0})
        if i["status"] == "waiting":
            a["waiting"].append({"item": i["item"], "qty": i["qty"], "urgency": i["urgency"],
                                 "waiting_min": (now - i["reported_at"]) // 60, "note": i["note"]})
        a["delivered_units"] += i["requested"] - i["qty"]
    return {"time": _ist(now), "operation_started": _ist(st["started"]), "figures": summary(st),
            "areas": list(areas.values()), "hazards": [h["label"] for h in st["hazards"]],
            "stock": {names.get(d, d): items for d, items in st["stock"].items()},
            "recent_waves": st["waves"][-3:]}


SITREP_PROMPT = """You are the duty officer of a Chennai flood-relief operation. Write a situation report for the
coordinator using ONLY the data below. Be specific (areas, items, quantities, minutes) and brief.
Return ONLY JSON: {"headline": "<one line>", "overview": "<2-3 sentences>",
"priorities": ["<most urgent unmet needs, max 5>"], "blocked_roads": ["..."], "stock_alerts": ["..."],
"actions": ["<concrete next steps, max 5>"]}
Data: """


def _rule_sitrep(data):
    f = data["figures"]
    waiting = sorted(((a["zone"], w) for a in data["areas"] for w in a["waiting"]),
                     key=lambda x: (-x[1]["urgency"], -x[1]["waiting_min"]))
    low = [(depot, item, qty) for depot, items in data["stock"].items() for item, qty in items.items() if qty < LOW_STOCK]
    actions = []
    if waiting:
        actions.append(f"Dispatch a wave for the {len(waiting)} waiting request(s), most urgent first.")
    actions += [f"Restock {item} at {depot}." for depot, item, _ in low[:2]]
    if data["hazards"]:
        actions.append(f"Verify and clear reported hazards: {', '.join(data['hazards'][:3])}.")
    return {"headline": f"{f['waiting']} request(s) waiting, {f['delivered']} delivered, {f['units_delivered']} units moved",
            "overview": (f"Operation running since {data['operation_started']} with {f['waves']} dispatch wave(s). "
                         f"Longest current wait is {f['longest_wait_min']} min. {f['hazards']} road hazard(s) on record."),
            "priorities": [f"{z}: {w['qty']} {w['item']}, urgency {w['urgency']}, waiting {w['waiting_min']} min"
                           for z, w in waiting[:5]],
            "blocked_roads": data["hazards"][:8],
            "stock_alerts": [f"{depot}: {item} at {qty} units" for depot, item, qty in low[:6]],
            "actions": actions[:5] or ["No action needed."]}


def _clean(out, fallback):
    def lst(v, n):
        return [str(x)[:200] for x in v][:n] if isinstance(v, list) else []
    return {"headline": str(out.get("headline") or fallback["headline"])[:160],
            "overview": str(out.get("overview") or fallback["overview"])[:600],
            "priorities": lst(out.get("priorities"), 5), "blocked_roads": lst(out.get("blocked_roads"), 8),
            "stock_alerts": lst(out.get("stock_alerts"), 6), "actions": lst(out.get("actions"), 5)}


def render(rep, data, generated_by):
    lines = [f"SITUATION REPORT - ReliefRoute - {data['time']}", "", rep["headline"], "", "Overview", rep["overview"]]
    for title, key in (("Priorities", "priorities"), ("Blocked roads", "blocked_roads"),
                       ("Stock alerts", "stock_alerts"), ("Recommended actions", "actions")):
        lines += ["", title] + ([f"- {x}" for x in rep[key]] or ["- none"])
    f = data["figures"]
    lines += ["", f"Figures: {f['waiting']} waiting, {f['delivered']} delivered, {f['units_delivered']} units delivered, "
                  f"{f['hazards']} hazards, {f['waves']} waves.",
              f"Generated by {'Gemini' if generated_by == 'gemini' else 'rule-based fallback'} from live operations data."]
    return "\n".join(lines)


def sitrep():
    heals = []
    data = _sitrep_data(load())
    fallback, rep, by = _rule_sitrep(data), None, "rules"
    if agent.llm_available():
        try:
            rep, by = _clean(agent.gemini_json([SITREP_PROMPT + json.dumps(data)], heals), fallback), "gemini"
        except Exception as e:
            heals.append(f"Situation report via Gemini failed ({str(e)[:60]}) -> rule-based report")
    rep = rep or fallback
    return {"generated_by": by, "report": rep, "text": render(rep, data, by), "heals": heals, "data": data}
