"""Live operations: the real dispatch loop.

    field report -> incident queue (duplicates merged; the reporter gets a tracking link)
    -> wave: the agent plans over the waiting incidents with the idle trucks and current stock, on
       real road distances that go around every reported hazard (roads.py)
    -> each truck with stops gets a route; stock is loaded and the truck is out until the route closes
    -> the driver opens the truck's link, follows the road geometry and marks every stop delivered,
       failed or road blocked
    -> outcomes flow back: failed and blocked stops return to the queue, a blocked road becomes a
       hazard that every route avoids and trucks already heading into it are rerouted from where
       they are (reroute_affected), undelivered cargo goes back into stock, and the agent learns
       its real ETA error, field results per strategy and field lessons (agent.record_field_route)
    -> autopilot.py decides by itself when the next wave is due (plan_wave now, commit_wave on approval)

Coordinators manage depots, trucks and incoming supplies at runtime.
State is one JSON document, "ops", in Firestore when configured (shared by every server instance) or
a local file; every change is an atomic read-modify-write through store.update.
"""
import copy
import json
import math
import os
import secrets
import time

import agent
import roads
import store
from env import DEPOTS, ITEMS, TRUCK_CAPACITY, TRUCKS_PER_DEPOT, ReliefEnv, _segment_hits, _xy, road_distances

SCHEMA = 2
OPS_STOCK = 400  # units of each item per depot when an operation starts
LOW_STOCK = 100
CHENNAI = (12.8, 13.35, 80.0, 80.4)  # lat min, lat max, lng min, lng max
SERVICE_MIN = 10  # unloading time per stop
DUP_KM = 0.5  # two GPS positions this close, same item, still open, within DUP_WINDOW_S -> the same need
DUP_GUESS_KM = 1.5  # when a position was guessed from a place name
DUP_WINDOW_S = 12 * 3600
KEEP_SHAPES = 20  # closed routes that keep their road geometry
BLOCKED_RADIUS_KM = 0.5
CONFIRM_KM = 1.0  # a second report of a hazard this close confirms the first
OUTCOMES = ("delivered", "failed", "blocked")
MAX_REROUTES = 5  # trucks rerouted per new hazard; the rest keep their route and see the hazard on the map
REROUTE_MARGIN_KM = 0.1
AUTOPILOT = {"mode": "off", "min_urgency": 4, "max_wait_min": 30, "dirty": None, "lease_until": 0,
             "last_plan_at": 0, "considered": [], "approved": 0}  # see autopilot.py


class OpsError(Exception):
    """A request that conflicts with the current operation (shown to the user as is)."""


def _path():
    return os.getenv("OPS_PATH") or ("/tmp/ops.json" if os.getenv("VERCEL") else "ops.json")


def _token():
    return secrets.token_urlsafe(12)


def _now():
    return int(time.time())


def _new_truck(truck_id, depot_id, capacity=TRUCK_CAPACITY):
    return {"id": truck_id, "depot_id": depot_id, "capacity": capacity, "status": "idle", "route_id": None,
            "token": _token()}


def new_state():
    return {"schema": SCHEMA, "version": 0, "started": _now(), "next_id": 1, "next_route": 1, "next_hazard": 1,
            "depots": [{"id": i, "name": n, "lat": la, "lng": lo, "stock": {it: OPS_STOCK for it in ITEMS}}
                       for i, n, la, lo in DEPOTS],
            "trucks": [_new_truck(f"{i}-T{k + 1}", i) for i, *_ in DEPOTS for k in range(TRUCKS_PER_DEPOT)],
            "incidents": [], "reports": [], "routes": [], "hazards": [], "waves": [], "restocks": [], "log": [],
            "autopilot": copy.deepcopy(AUTOPILOT), "proposals": []}


def _fill(st):
    """Keys added after schema 2 shipped get their defaults, so a running operation is never wiped."""
    st.setdefault("proposals", [])
    ap = st.setdefault("autopilot", {})
    for k, v in AUTOPILOT.items():
        ap.setdefault(k, copy.deepcopy(v))
    return st


def _update(mutate):
    """Atomic read-modify-write of the operation; mutate(state) edits in place and returns a value.
    Raising OpsError inside mutate aborts without writing."""
    def guarded(st):
        if st.get("schema") != SCHEMA:  # first use, or a document from an older version
            version = st.get("version", 0)
            st.clear()
            st.update(new_state(), version=version)
        out = mutate(_fill(st))
        st["version"] += 1
        return out
    return store.update("ops", _path(), new_state, guarded)


def load():
    st = store.read("ops", _path(), new_state)
    return _fill(st) if st.get("schema") == SCHEMA else _update(lambda s: s)


def reset(actor="console"):
    def fresh(st):
        version = st["version"]  # keep counting so the version never goes backwards
        st.clear()
        st.update(new_state(), version=version)
        _log(st, actor, "Started a new operation")
        return st
    return _update(fresh)


# ---- helpers -------------------------------------------------------------------------------
def _in_chennai(lat, lng):
    return CHENNAI[0] < lat < CHENNAI[1] and CHENNAI[2] < lng < CHENNAI[3]


def _km(a, b):
    (x1, y1), (x2, y2) = _xy(*a), _xy(*b)
    return math.hypot(x2 - x1, y2 - y1)


def _log(st, actor, text):
    st["log"] = (st["log"] + [{"at": _now(), "by": actor or "console", "text": text}])[-200:]


def _set_status(inc):
    if inc["closed"]:
        inc["status"] = "closed"
    elif inc["qty"] > 0:
        inc["status"] = "waiting"  # may also have part of it on a truck (assigned > 0)
    elif inc["assigned"] > 0:
        inc["status"] = "en_route"
    else:
        inc["status"] = "delivered"


def _find(items, key, value):
    return next((x for x in items if x[key] == value), None)


def _eta_at(route, stop):
    """Expected arrival as a time (stops planned before eta_at existed only have minutes after dispatch)."""
    return stop.get("eta_at") or route["created_at"] + stop["eta_min"] * 60


def _active_hazards(st):
    return [h for h in st["hazards"] if h["active"]]


def _routing_hazards(st):
    """Hazards routes go around: confirmed ones. One unconfirmed report must not wall off a road."""
    return [h for h in st["hazards"] if h["active"] and h.get("verified", True)]


def _add_hazard(st, label, lat, lng, radius_km, source, verified=True):
    h = {"id": f"H{st['next_hazard']}", "label": str(label)[:80], "lat": lat, "lng": lng,
         "radius_km": radius_km, "source": source, "reported_at": _now(), "active": True,
         "verified": verified, "confirmations": 1}
    st["next_hazard"] += 1
    st["hazards"].append(h)
    return h


def _dirty(st, why):
    """Something changed that may make a new wave worth planning (autopilot.py picks it up)."""
    st["autopilot"]["dirty"] = str(why)[:120]


# ---- field reports ----------------------------------------------------------------------------
def _duplicate_of(st, item, zone, lat, lng, gps, now, guessed=False):
    """An open incident for the same item at the same place, reported recently: the same need reported
    twice. Two GPS positions must be within DUP_KM; a position guessed from a place name is only as
    precise as the locality, so then the same locality (or DUP_GUESS_KM) is enough. A request with no
    known place is never merged: "somewhere in Chennai" is not the same place twice."""
    if guessed:
        return None

    def same_place(i, km):
        if gps and i["located_by"] == "gps":
            return km <= DUP_KM
        return km <= DUP_GUESS_KM or i["zone"].strip().lower() == str(zone).strip().lower()
    near = [(_km((lat, lng), (i["lat"], i["lng"])), i) for i in st["incidents"]
            if i["item"] == item and not i["closed"] and (i["qty"] > 0 or i["assigned"] > 0)
            and i["located_by"] != "guess" and now - i["reported_at"] <= DUP_WINDOW_S]
    near = [x for x in near if same_place(x[1], x[0])]
    return min(near, key=lambda x: x[0])[1] if near else None


def add_report(report, source="console", coords=None, actor=None):
    """Turn an intake result into incidents (or merge it into existing ones) and give the reporter a
    tracking token. A phone's GPS position overrides guessed coordinates.

    A hazard from a report starts unconfirmed: it is shown, but routes only avoid it once a second
    report, a driver or a coordinator confirms it. Returns the ids of hazards this report confirmed."""
    gps = coords if coords and _in_chennai(*coords) else None
    now, track = _now(), _token()

    def add(st):
        made, merged = [], []
        for n in report["needs"]:
            lat, lng = gps if gps else (n["lat"], n["lng"])
            located = "gps" if gps else n.get("located", "report")
            dup = _duplicate_of(st, n["item"], n["zone"], lat, lng, bool(gps), now, guessed=located == "guess")
            if dup:  # the larger figure wins: two people describing one need must not send two trucks
                extra = max(0, n["qty"] - dup["requested"])
                dup.update(requested=dup["requested"] + extra, qty=dup["qty"] + extra,
                           urgency=max(dup["urgency"], n["urgency"]), reports=dup["reports"] + 1)
                if gps and dup["located_by"] != "gps":
                    dup.update(lat=lat, lng=lng, located_by="gps")
                _set_status(dup)
                merged.append(dup)
                continue
            inc = {"id": f"I{st['next_id']}", "zone": n["zone"], "lat": lat, "lng": lng, "item": n["item"],
                   "requested": n["qty"], "qty": n["qty"], "assigned": 0, "delivered": 0, "urgency": n["urgency"],
                   "status": "waiting", "closed": False, "source": source, "reports": 1,
                   "summary": str(report.get("summary", ""))[:200], "evidence": str(n.get("evidence", ""))[:200],
                   "note": "location unknown: check before dispatching" if located == "guess" else "",
                   "located_by": located, "reported_at": now, "delivered_at": None}
            st["next_id"] += 1
            st["incidents"].append(inc)
            made.append(inc)
        hazards, confirmed = [], []
        for h in report.get("hazards", []):
            if not _in_chennai(h["lat"], h["lng"]):
                continue
            near = next((x for x in _active_hazards(st) if _km((x["lat"], x["lng"]), (h["lat"], h["lng"])) <= CONFIRM_KM), None)
            if near:  # the same hazard reported again
                near["confirmations"] = near.get("confirmations", 1) + 1
                if not near.get("verified", True):
                    near["verified"] = True
                    confirmed.append(near["id"])
                    _log(st, actor or source, f"{near['id']} confirmed by a second report: routes now avoid it")
                continue
            hazards.append(_add_hazard(st, h.get("label", "Reported hazard"), h["lat"], h["lng"], 0.8, source,
                                       verified=False))
        st["reports"] = (st["reports"] + [{"token": track, "at": now, "source": source,
                                           "summary": str(report.get("summary", ""))[:200],
                                           "incidents": [i["id"] for i in made + merged],
                                           "hazards": len(hazards) + len(confirmed)}])[-500:]
        parts = [f"{len(made)} new"] + ([f"{len(merged)} merged into {', '.join(i['id'] for i in merged)}"] if merged else [])
        parts += [f"{len(hazards)} unconfirmed hazard(s)"] if hazards else []
        _log(st, actor or source, f"Report from {source}: {' · '.join(parts)}")
        return {"incidents": made, "merged": merged, "track": track, "confirmed": confirmed,
                "hazards": [h["id"] for h in hazards], "ops": st}
    return _update(add)


def track_view(token):
    """What a reporter may see about their own report."""
    st = load()
    rep = next((r for r in st["reports"] if secrets.compare_digest(r["token"], token)), None)
    if not rep:
        return None
    out = []
    for iid in rep["incidents"]:
        inc = _find(st["incidents"], "id", iid)
        if not inc:
            continue
        trips = [{"truck": r["truck_id"], "qty": s["qty"], "status": s["status"], "at": s["at"],
                  "eta_at": _eta_at(r, s) if s["status"] == "pending" else None}
                 for r in st["routes"] for s in r["stops"] if s["incident_id"] == iid]
        out.append({k: inc[k] for k in ("id", "zone", "item", "requested", "qty", "assigned", "delivered", "urgency",
                                        "status", "reports", "reported_at", "delivered_at")} | {"trips": trips})
    return {"report": {k: rep[k] for k in ("at", "source", "summary", "hazards")}, "incidents": out}


# ---- dispatch waves --------------------------------------------------------------------------
def _route_for(points, hazards, label, heals):
    try:
        r = roads.route(points, hazards)
        if r:
            return r
    except Exception as e:
        heals.append(f"Road route for {label} failed ({str(e)[:70]}) -> straight-line estimate")
    return roads.estimate(points)


def plan_wave(st, heals, skip_unlocated=False):
    """Plan one wave over the waiting incidents with the idle trucks. Writes nothing: commit_wave sends
    the trucks out, now (a coordinator's wave) or later (an autopilot proposal someone approves).
    skip_unlocated leaves out requests with no known place (the autopilot never guesses an address)."""
    waiting = [i for i in st["incidents"] if i["qty"] > 0 and not i["closed"]
               and not (skip_unlocated and i["located_by"] == "guess")]
    idle = [t for t in st["trucks"] if t["status"] == "idle"]
    if not waiting:
        raise OpsError("No waiting incidents to dispatch.")
    if not idle:
        raise OpsError("Every truck is out on a route. Wait for a driver to finish, recall a route, or add a truck.")
    hazards = _routing_hazards(st)
    depots = [{**d, "trucks": [t["id"] for t in idle if t["depot_id"] == d["id"]],
               "capacity": {t["id"]: t["capacity"] for t in idle if t["depot_id"] == d["id"]}} for d in st["depots"]]
    needs = [{k: i[k] for k in ("id", "zone", "lat", "lng", "item", "qty", "requested", "urgency")} for i in waiting]

    road = None
    try:
        road = roads.table([(d["lat"], d["lng"]) for d in depots if d["trucks"]] +
                           [(n["lat"], n["lng"]) for n in needs], hazards)
        if road is None and roads.enabled():
            heals.append("Too many stops for one road-distance request -> straight-line estimates for planning")
    except Exception as e:
        heals.append(f"Road router unavailable ({str(e)[:70]}) -> straight-line estimates for planning")

    mem = agent.load_memory()
    choice, factor = agent.choose_live_strategy(mem), mem["eta_factor"]
    strategy = choice["strategy"]
    with road_distances(road):
        env = ReliefEnv()
        obs = env.reset(n_needs=0, chaos=False, random_hazards=False, extra_hazards=hazards, extra_needs=needs,
                        depots=depots)
        action, why = agent.plan_round(obs, strategy, mem["lessons"], heals)
        info = env.step({"assignments": action})["info"]

    incs = {i["id"]: i for i in waiting}
    plans = []
    for tid in dict.fromkeys(d["truck_id"] for d in info["delivered"]):
        legs = sorted((d for d in info["delivered"] if d["truck_id"] == tid), key=lambda d: d["stop"])
        depot = _find(st["depots"], "id", legs[0]["depot_id"])
        points = [(depot["lat"], depot["lng"])] + [(incs[d["need_id"]]["lat"], incs[d["need_id"]]["lng"]) for d in legs]
        r = _route_for(points, hazards, tid, heals)
        elapsed, stops = 0.0, []
        for k, (d, leg) in enumerate(zip(legs, r["legs"])):
            elapsed += leg["minutes"] + (SERVICE_MIN if k else 0)  # unloading at each earlier stop
            inc = incs[d["need_id"]]
            stops.append({"n": k + 1, "incident_id": inc["id"], "zone": inc["zone"], "item": d["item"], "qty": d["qty"],
                          "lat": inc["lat"], "lng": inc["lng"], "km": leg["km"], "eta_raw": round(elapsed, 1),
                          "eta_min": round(elapsed * factor), "status": "pending", "delivered_qty": 0, "note": "",
                          "at": None})
        plans.append({"truck_id": tid, "depot_id": depot["id"], "km": r["km"], "minutes": round(r["minutes"] * factor),
                      "source": r["source"], "shape": r["shape"], "stops": stops})
    return {"plans": plans, "info": {k: info[k] for k in ("blocked", "no_truck", "stockout")},
            "strategy": strategy, "reasoning": why, "strategy_why": choice["why"],
            "scores": {s: v["expected"] for s, v in choice["scores"].items()}, "eta_factor": factor,
            "road": road is not None, "needs": [n["id"] for n in needs], "hazards": [h["id"] for h in hazards]}


def commit_wave(planned, actor, heals, hook=None):
    """Send out the trucks of a planned wave, re-validated on fresh data: reports, drivers and other
    waves may have changed things since it was planned. hook(st, wave) runs in the same atomic write
    (raising OpsError there cancels the whole wave)."""
    plans, strategy, factor, now = planned["plans"], planned["strategy"], planned["eta_factor"], _now()

    def apply(st):
        wave_no = len(st["waves"]) + 1
        trucks = {t["id"]: t for t in st["trucks"]}
        live = {i["id"]: i for i in st["incidents"]}
        made = []
        for p in plans:
            t = trucks.get(p["truck_id"])
            if not t or t["status"] != "idle":
                heals.append(f"{p['truck_id']} became busy while this wave was planned -> its stops stay queued")
                continue
            route_id = f"R{st['next_route']}"
            stock = _find(st["depots"], "id", p["depot_id"])["stock"]
            kept = []
            for s in p["stops"]:
                inc = live.get(s["incident_id"])
                q = min(s["qty"], inc["qty"] if inc and not inc["closed"] else 0, stock.get(s["item"], 0))
                if q <= 0:
                    continue
                stock[s["item"]] -= q
                inc["qty"] -= q
                inc["assigned"] += q
                inc["note"] = f"{route_id}: {t['id']} stop {s['n']}, ETA {s['eta_min']} min"
                _set_status(inc)
                kept.append({**s, "qty": q, "eta_at": now + s["eta_min"] * 60})
            if not kept:
                continue
            st["next_route"] += 1
            route = {"id": route_id, "wave": wave_no, "truck_id": t["id"], "depot_id": p["depot_id"], "status": "active",
                     "created_at": now, "closed_at": None, "strategy": strategy, "km": p["km"], "minutes": p["minutes"],
                     "source": p["source"], "shape": p["shape"], "eta_factor": factor, "stops": kept,
                     "rev": 1, "reroutes": [], "warning": None, "last_pos": None}
            t.update(status="on_route", route_id=route_id)
            st["routes"].append(route)
            made.append(route)
        for b in planned["info"]["blocked"]:
            if b["need_id"] in live:
                live[b["need_id"]]["note"] = f"wave {wave_no}: no road route around {', '.join(b['hazards'])}; retry next wave"
        for b in planned["info"]["no_truck"] + planned["info"]["stockout"]:
            inc = live.get(b["need_id"])
            if inc and inc["qty"] > 0:
                inc["note"] = f"wave {wave_no}: no free truck or stock at {b['depot_id']}; retry next wave"
        _prune_shapes(st)
        wave = {"n": wave_no, "at": now, "strategy": strategy, "reasoning": planned["reasoning"],
                "strategy_why": planned.get("strategy_why", ""), "scores": planned.get("scores", {}),
                "open_before": len(planned["needs"]), "routes": len(made), "stops": sum(len(r["stops"]) for r in made),
                "units": sum(s["qty"] for r in made for s in r["stops"]),
                "left_waiting": sum(1 for i in st["incidents"] if i["qty"] > 0 and not i["closed"]),
                "road": planned["road"], "by": actor or "console"}
        if hook:
            hook(st, wave)
        st["waves"].append(wave)
        _log(st, actor, f"Wave {wave_no}: {wave['routes']} route(s), {wave['units']} units, strategy {strategy}")
        return {"wave": wave, "routes": made, "ops": st}
    out = _update(apply)
    if not out["routes"]:
        heals.append("No truck could take any stop this wave (stock or capacity) -> incidents stay queued")
    return {**out, "strategy": strategy, "reasoning": planned["reasoning"], "strategy_why": planned.get("strategy_why", ""),
            "heals": heals, "plan": planned["info"]}


def run_wave(actor="console"):
    """Plan one wave over every waiting incident with the idle trucks, then send each truck out."""
    heals = []
    return commit_wave(plan_wave(load(), heals), actor, heals)


def _prune_shapes(st):
    closed = [r for r in st["routes"] if r["status"] != "active" and r.get("shape")]
    for r in closed[:-KEEP_SHAPES]:
        r["shape"] = []


# ---- routes and drivers ---------------------------------------------------------------------
def _close(st, route, truck, why, actor):
    """End a route: unvisited stops go back to the queue, leftover cargo back into stock, the truck
    becomes free. Returns the outcome the agent learns from."""
    now = _now()
    incs = {i["id"]: i for i in st["incidents"]}
    for s in route["stops"]:
        if s["status"] == "pending":
            s.update(status="skipped", at=now, note=why)
            inc = incs.get(s["incident_id"])
            if inc:
                inc["assigned"] = max(0, inc["assigned"] - s["qty"])
                inc["qty"] += s["qty"]
                inc["note"] = f"{route['id']}: not reached ({why}); back in queue"
                _set_status(inc)
    stock = _find(st["depots"], "id", route["depot_id"])["stock"]
    for s in route["stops"]:
        stock[s["item"]] = stock.get(s["item"], 0) + s["qty"] - s["delivered_qty"]
    route.update(status="done", closed_at=now)
    truck.update(status="idle", route_id=None)
    if any(i["qty"] > 0 and not i["closed"] for i in st["incidents"]):
        _dirty(st, f"{truck['id']} is free again")
    _log(st, actor, f"{route['id']} closed ({why}): "
                    f"{sum(s['status'] == 'delivered' for s in route['stops'])}/{len(route['stops'])} stops delivered")
    return {"route": route["id"], "strategy": route["strategy"],
            "stops": [{"zone": s["zone"], "urgency": incs[s["incident_id"]]["urgency"] if s["incident_id"] in incs else 3,
                       "qty": s["qty"], "delivered_qty": s["delivered_qty"], "status": s["status"], "note": s["note"],
                       "eta_raw": s["eta_raw"], "actual_min": round(((s["at"] or now) - route["created_at"]) / 60, 1)}
                      for s in route["stops"]]}


def _learn(outcome):
    try:
        return agent.record_field_route(outcome)
    except Exception as e:  # the route is closed either way; report that this outcome was not learned
        return {"error": f"Could not record field outcome ({str(e)[:60]})"}


def _by_token(st, token):
    return next((t for t in st["trucks"] if secrets.compare_digest(t["token"], token)), None)


def driver_view(token):
    st = load()
    t = _by_token(st, token)
    if not t:
        return None
    route = _find(st["routes"], "id", t["route_id"]) if t["route_id"] else None
    depot = _find(st["depots"], "id", t["depot_id"])
    return {"truck": {k: t[k] for k in ("id", "capacity", "status", "depot_id")},
            "depot": {k: depot[k] for k in ("id", "name", "lat", "lng")}, "route": route,
            "hazards": _active_hazards(st)}


def driver_update(token, stop_n, outcome, qty=None, note="", coords=None):
    """A driver reports what happened at one stop. Closing the last stop closes the route."""
    if outcome not in OUTCOMES:
        raise OpsError(f"Outcome must be one of {', '.join(OUTCOMES)}.")
    note = str(note or "")[:120]

    def upd(st):
        t = _by_token(st, token)
        route = _find(st["routes"], "id", t["route_id"]) if t and t["route_id"] else None
        if not route:
            raise OpsError("This truck has no active route.")
        stop = next((s for s in route["stops"] if s["n"] == stop_n), None)
        if not stop or stop["status"] != "pending":
            raise OpsError("That stop is already recorded.")
        inc = _find(st["incidents"], "id", stop["incident_id"])
        now = _now()
        gps = coords if coords and _in_chennai(*coords) else None
        if gps:  # where the truck is: rerouting starts here
            route["last_pos"] = [round(gps[0], 5), round(gps[1], 5)]
        hazard = None
        if outcome == "delivered":
            dq = stop["qty"] if qty is None else max(0, min(stop["qty"], int(qty)))
            stop.update(status="delivered", delivered_qty=dq, at=now, note=note)
            if inc:
                inc["assigned"] = max(0, inc["assigned"] - stop["qty"])
                inc["delivered"] += dq
                inc["qty"] += stop["qty"] - dq  # a short delivery leaves the rest in the queue
                inc["note"] = f"{route['id']}: {dq} delivered by {t['id']}" + \
                              (f", {stop['qty'] - dq} still needed" if dq < stop["qty"] else "")
        else:
            stop.update(status=outcome, at=now, note=note)
            if inc:
                inc["assigned"] = max(0, inc["assigned"] - stop["qty"])
                inc["qty"] += stop["qty"]
                inc["note"] = f"{route['id']}: {'road blocked' if outcome == 'blocked' else 'delivery failed'}" \
                              f"{': ' + note if note else ''}; back in queue"
            if outcome == "blocked":  # the blockage is where the driver is, else on the way to the stop
                lat, lng = gps or _midpoint(_last_reached(st, route), (stop["lat"], stop["lng"]))
                hazard = _add_hazard(st, f"Road blocked: {note or 'reported by driver'} ({t['id']})", lat, lng,
                                     BLOCKED_RADIUS_KM, "driver")
                _log(st, f"driver {t['id']}", f"Reported {hazard['id']} near {stop['zone']}")
            _dirty(st, f"{t['id']} stop {stop['n']} {stop['zone']}: {outcome}")
        if inc:
            if inc["qty"] == 0 and inc["assigned"] == 0 and not inc["delivered_at"]:
                inc["delivered_at"] = now
            _set_status(inc)
        _log(st, f"driver {t['id']}", f"{route['id']} stop {stop['n']} ({stop['zone']}): {outcome}")
        closed = None
        if all(s["status"] != "pending" for s in route["stops"]):
            closed = _close(st, route, t, "all stops done", f"driver {t['id']}")
        return {"stop": stop, "closed": closed, "hazard": hazard["id"] if hazard else None}
    out = _update(upd)
    out["learned"] = _learn(out["closed"]) if out["closed"] else None
    return out


def _last_reached(st, route):
    """The last place the truck is known to have been: its latest recorded stop, else the depot."""
    reached = [s for s in route["stops"] if s["status"] in ("delivered", "failed") and s["at"]]
    if reached:
        s = max(reached, key=lambda s: (s["at"], s["n"]))
        return s["lat"], s["lng"]
    d = _find(st["depots"], "id", route["depot_id"])
    return d["lat"], d["lng"]


def _midpoint(a, b):
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


# ---- rerouting trucks already on the road -----------------------------------------------------------
def _inside(p, hazards):
    return any(_km(p, (h["lat"], h["lng"])) <= h["radius_km"] for h in hazards)


def _origin(st, route, hazards):
    """Where the truck drives on from: its last GPS position, else its last stop, else the depot. A point
    inside an avoided area is passed over: the router cannot start there, and the truck has to turn back
    out of it anyway."""
    options = ([tuple(route["last_pos"])] if route.get("last_pos") else []) + [_last_reached(st, route)]
    return next((p for p in options if not _inside(p, hazards)), options[-1])


def _crosses(route, origin, pending, h):
    """Does the rest of this route pass through hazard h? "stop" when h covers a stop itself: no road can
    avoid that, so the driver is warned instead."""
    centre = (h["lat"], h["lng"])
    if any(_km((s["lat"], s["lng"]), centre) <= h["radius_km"] for s in pending):
        return "stop"
    reach = h["radius_km"] + REROUTE_MARGIN_KM
    shape = route.get("shape") or []
    if route["source"] == "road" and len(shape) > 1:
        i = min(range(len(shape)), key=lambda k: _km(origin, shape[k]))  # only the part still ahead
        return any(_km(p, centre) <= reach for p in shape[i:])
    pts = [origin] + [(s["lat"], s["lng"]) for s in pending]
    return any(_segment_hits(a, b, {**h, "radius_km": reach}) for a, b in zip(pts, pts[1:]))


def _replan(origin, pending, hazards):
    """A road route from the truck through its remaining stops around every hazard. Stops no road reaches
    any more are dropped. {"route", "keep", "drop"}, or {"error"} when the router cannot answer."""
    def go(stops):
        return roads.route([origin] + [(s["lat"], s["lng"]) for s in stops], hazards)
    try:
        r = go(pending)
        return {"route": r, "keep": pending, "drop": []} if r else {"error": "road routing is switched off"}
    except roads.NoRoute:
        pass
    except Exception as e:
        return {"error": str(e)[:70]}
    keep, drop = [], []
    try:
        for s in pending:  # which stops are cut off?
            try:
                go([s])
                keep.append(s)
            except roads.NoRoute:
                drop.append(s)
        return {"route": go(keep) if keep else None, "keep": keep, "drop": drop}
    except Exception as e:
        return {"error": str(e)[:70]}


def _commit_reroute(route_id, rev, pending_ns, plan, crossing, at_stop, actor):
    now = _now()

    def upd(st):
        route = _find(st["routes"], "id", route_id)
        if not route or route["status"] != "active" or route.get("rev", 1) != rev or \
                [s["n"] for s in route["stops"] if s["status"] == "pending"] != pending_ns:
            raise OpsError(f"{route_id} changed while a new route was computed; it keeps its current one")
        warn = f"{', '.join(at_stop)} reported at one of your stops: approach with care." if at_stop else None
        if not crossing:
            route["warning"] = warn
            _log(st, actor, f"{route_id}: driver warned about {', '.join(at_stop)} at a stop")
            return {"route": route_id, "changed": False, "warning": warn, "closed": None}
        if "error" in plan:
            route["warning"] = f"{', '.join(crossing)} reported on your way and no new route could be computed: " \
                               f"drive around it."
            _log(st, actor, f"{route_id}: could not reroute around {', '.join(crossing)} ({plan['error']}); driver warned")
            return {"route": route_id, "changed": False, "warning": route["warning"], "closed": None,
                    "heal": f"Reroute of {route_id} failed ({plan['error']}) -> driver warned, route unchanged"}
        pending = {s["n"]: s for s in route["stops"] if s["status"] == "pending"}
        km_before = round(sum(s["km"] for s in pending.values()), 1)
        incs = {i["id"]: i for i in st["incidents"]}
        dropped = [d["n"] for d in plan["drop"]]
        for n in dropped:
            s = pending[n]
            s.update(status="unreachable", at=now, note=f"no road around {', '.join(crossing)}")
            inc = incs.get(s["incident_id"])
            if inc:  # the cargo stays on the truck and goes back into stock when the route closes
                inc["assigned"] = max(0, inc["assigned"] - s["qty"])
                inc["qty"] += s["qty"]
                inc["note"] = f"{route_id}: no road around {', '.join(crossing)}; back in queue"
                _set_status(inc)
        r, elapsed = plan["route"], 0.0
        for k, (d, leg) in enumerate(zip(plan["keep"], r["legs"] if r else [])):
            elapsed += leg["minutes"] + (SERVICE_MIN if k else 0)
            s = pending[d["n"]]
            s.update(km=leg["km"], eta_raw=round((now - route["created_at"]) / 60 + elapsed, 1),
                     eta_at=now + round(elapsed * route["eta_factor"] * 60))
            s["eta_min"] = round((s["eta_at"] - route["created_at"]) / 60)
        km_after = r["km"] if r else 0.0
        if r:
            route.update(shape=r["shape"], source=r["source"])
        route["km"] = round(sum(s["km"] for s in route["stops"] if s["status"] != "unreachable"), 1)
        left = [s for s in route["stops"] if s["status"] == "pending"]
        if left:
            route["minutes"] = round((max(s["eta_at"] for s in left) - route["created_at"]) / 60)
        route["rev"] = rev + 1
        route["reroutes"] = (route.get("reroutes", []) + [{"at": now, "hazards": crossing, "km_before": km_before,
                                                           "km_after": km_after, "dropped": dropped}])[-10:]
        route["warning"] = warn
        _log(st, actor, f"Rerouted {route_id} around {', '.join(crossing)}: {km_before} -> {km_after} km left"
                        + (f"; stop(s) {', '.join(map(str, dropped))} cut off, back in queue" if dropped else ""))
        if dropped:
            _dirty(st, f"{route_id}: {len(dropped)} stop(s) cut off by {', '.join(crossing)}")
        closed = None
        if not left:
            closed = _close(st, route, _find(st["trucks"], "id", route["truck_id"]), "no road to the remaining stops", actor)
        return {"route": route_id, "changed": True, "km_before": km_before, "km_after": km_after,
                "dropped": dropped, "closed": closed}
    return _update(upd)


def reroute_affected(hazard_ids, actor="agent"):
    """After a road is confirmed blocked, every truck on the road whose remaining route crosses it gets a
    new road route around it, starting where it is; stops no road reaches any more go back to the queue.
    Returns {"rerouted": [per route], "heals": [...], "learned": [outcomes of routes that ended]}."""
    out = {"rerouted": [], "heals": [], "learned": []}
    if not hazard_ids:
        return out
    st = load()
    hazards = _routing_hazards(st)
    new = [h for h in hazards if h["id"] in hazard_ids]
    for route in [r for r in st["routes"] if r["status"] == "active"]:
        pending = [s for s in route["stops"] if s["status"] == "pending"]
        if not pending or not new:
            continue
        origin = _origin(st, route, hazards)
        hits = {h["id"]: _crosses(route, origin, pending, h) for h in new}
        crossing = [k for k, v in hits.items() if v is True]
        at_stop = [k for k, v in hits.items() if v == "stop"]
        if not crossing and not at_stop:
            continue
        if crossing and len(out["rerouted"]) >= MAX_REROUTES:
            out["heals"].append(f"More than {MAX_REROUTES} routes cross {', '.join(crossing)} -> {route['id']} keeps "
                                f"its route; its driver sees the hazard on the map")
            continue
        plan = _replan(origin, pending, hazards) if crossing else {}
        try:
            res = _commit_reroute(route["id"], route.get("rev", 1), [s["n"] for s in pending], plan, crossing,
                                  at_stop, actor)
        except OpsError as e:
            out["heals"].append(str(e))
            continue
        if res.get("heal"):
            out["heals"].append(res.pop("heal"))
        if res["closed"]:
            out["learned"].append(_learn(res["closed"]))
        out["rerouted"].append(res)
    return out


def recall_route(route_id, actor="console"):
    """A coordinator ends a route early (truck broke down, driver unreachable)."""
    def rec(st):
        route = _find(st["routes"], "id", route_id)
        if not route or route["status"] != "active":
            raise OpsError(f"{route_id} is not an active route.")
        return _close(st, route, _find(st["trucks"], "id", route["truck_id"]), "recalled by coordinator", actor)
    outcome = _update(rec)
    return {"closed": outcome, "learned": _learn(outcome)}


# ---- resources -------------------------------------------------------------------------------------
def _capacity(capacity):
    try:
        capacity = int(capacity)
    except (TypeError, ValueError):
        capacity = 0
    if not 20 <= capacity <= 2000:
        raise OpsError("Truck capacity must be between 20 and 2000 units.")
    return capacity


def add_depot(name, lat, lng, actor="console"):
    name = str(name).strip()[:40]
    if not name or not _in_chennai(lat, lng):
        raise OpsError("A depot needs a name and a location inside the Chennai operation area.")

    def add(st):
        n = max([int(d["id"][1:]) for d in st["depots"] if d["id"][1:].isdigit()] + [0]) + 1
        d = {"id": f"D{n}", "name": name, "lat": lat, "lng": lng, "stock": {it: 0 for it in ITEMS}}
        st["depots"].append(d)
        _log(st, actor, f"Opened depot {d['id']} {name}")
        return d
    return _update(add)


def add_truck(depot_id, capacity, actor="console"):
    capacity = _capacity(capacity)

    def add(st):
        if not _find(st["depots"], "id", depot_id):
            raise OpsError(f"No depot {depot_id}.")
        n = sum(t["depot_id"] == depot_id for t in st["trucks"]) + 1
        while _find(st["trucks"], "id", f"{depot_id}-T{n}"):
            n += 1
        t = _new_truck(f"{depot_id}-T{n}", depot_id, capacity)
        st["trucks"].append(t)
        _log(st, actor, f"Added truck {t['id']} ({capacity} units)")
        return t
    return _update(add)


def set_truck(truck_id, capacity=None, in_service=None, actor="console"):
    capacity = None if capacity is None else _capacity(capacity)

    def upd(st):
        t = _find(st["trucks"], "id", truck_id)
        if not t:
            raise OpsError(f"No truck {truck_id}.")
        if capacity is not None:
            t["capacity"] = capacity
        if in_service is not None:
            if t["status"] == "on_route":
                raise OpsError(f"{truck_id} is on a route; recall it first.")
            t["status"] = "idle" if in_service else "out_of_service"
        _log(st, actor, f"Updated truck {truck_id}: {t['capacity']} units, {t['status'].replace('_', ' ')}")
        return t
    return _update(upd)


def restock(depot_id, item, qty, note="", actor="console"):
    try:
        qty = int(qty)
    except (TypeError, ValueError):
        qty = 0
    if item not in ITEMS or not 1 <= qty <= 100_000:
        raise OpsError(f"Restock needs an item ({', '.join(ITEMS)}) and a quantity between 1 and 100000.")

    def add(st):
        d = _find(st["depots"], "id", depot_id)
        if not d:
            raise OpsError(f"No depot {depot_id}.")
        d["stock"][item] = d["stock"].get(item, 0) + qty
        st["restocks"] = (st["restocks"] + [{"at": _now(), "depot_id": depot_id, "item": item, "qty": qty,
                                             "note": str(note)[:80], "by": actor or "console"}])[-200:]
        _log(st, actor, f"Restocked {depot_id}: +{qty} {item}" + (f" ({note})" if note else ""))
        return d
    return _update(add)


def confirm_hazard(hazard_id, actor="console"):
    """A coordinator confirms a reported hazard: from now on routes avoid it (main reroutes trucks)."""
    def upd(st):
        h = _find(st["hazards"], "id", hazard_id)
        if not h or not h["active"] or h.get("verified", True):
            raise OpsError(f"{hazard_id} is not an unconfirmed hazard.")
        h["verified"] = True
        _log(st, actor, f"Confirmed {hazard_id} ({h['label']}): routes now avoid it")
        return h
    return _update(upd)


def clear_hazard(hazard_id, actor="console"):
    def upd(st):
        h = _find(st["hazards"], "id", hazard_id)
        if not h or not h["active"]:
            raise OpsError(f"{hazard_id} is not an active hazard.")
        h["active"] = False
        _log(st, actor, f"Cleared {hazard_id} ({h['label']}): road reopened")
        return h
    return _update(upd)


def close_incident(incident_id, reason="", actor="console"):
    """A coordinator closes a request that needs no delivery (resolved locally, false report)."""
    reason = str(reason or "")[:80]

    def upd(st):
        inc = _find(st["incidents"], "id", incident_id)
        if not inc or inc["status"] in ("closed", "delivered"):
            raise OpsError(f"{incident_id} is not open.")
        if inc["assigned"]:
            raise OpsError(f"{incident_id} is on a truck; record or recall that route first.")
        inc.update(closed=True, qty=0, note=f"closed: {reason or 'by coordinator'}")
        _set_status(inc)
        _log(st, actor, f"Closed {incident_id}: {reason or 'no reason given'}")
        return inc
    return _update(upd)


# ---- views -------------------------------------------------------------------------------------
def _coarse(p):
    return round(p, 3)  # about 100 m: the area on the map, not a family's doorstep


def view(st, coordinator):
    """The operation as a client may see it. Tracking links never; for the public also no driver links,
    no reporters' words, positions rounded to ~100 m and coordinators' email addresses hidden."""
    out = copy.deepcopy(st)
    out["reports"] = [{k: r[k] for k in ("at", "source", "incidents")} for r in out["reports"][-50:]]
    if coordinator:
        return out
    for t in out["trucks"]:
        t.pop("token", None)
    for i in out["incidents"]:
        i.update(lat=_coarse(i["lat"]), lng=_coarse(i["lng"]), summary="", evidence="")
    for r in out["routes"]:
        r["last_pos"] = None
        r["shape"] = [[_coarse(a), _coarse(b)] for a, b in r["shape"]]
        for s in r["stops"]:
            s.update(lat=_coarse(s["lat"]), lng=_coarse(s["lng"]))
    for rows in ("log", "waves", "restocks", "proposals"):
        for x in out.get(rows, []):
            if "@" in str(x.get("by") or ""):
                x["by"] = "coordinator"
    for p in out.get("proposals", []):
        for plan in p.get("plans", []):
            plan["shape"] = [[_coarse(a), _coarse(b)] for a, b in plan.get("shape", [])]
            for s in plan["stops"]:
                s.update(lat=_coarse(s["lat"]), lng=_coarse(s["lng"]))
    return out


def summary(st):
    now = _now()
    waiting = [i for i in st["incidents"] if i["qty"] > 0 and not i["closed"]]
    return {"waiting": len(waiting), "en_route": sum(i["assigned"] > 0 for i in st["incidents"]),
            "delivered": sum(i["status"] == "delivered" for i in st["incidents"]),
            "units_delivered": sum(i["delivered"] for i in st["incidents"]),
            "hazards": len(_active_hazards(st)), "unconfirmed_hazards": len(_active_hazards(st)) - len(_routing_hazards(st)),
            "waves": len(st["waves"]),
            "trucks_out": sum(t["status"] == "on_route" for t in st["trucks"]),
            "trucks_free": sum(t["status"] == "idle" for t in st["trucks"]),
            "longest_wait_min": max([(now - i["reported_at"]) // 60 for i in waiting] or [0])}


# ---- situation report --------------------------------------------------------------
def _ist(ts):
    return time.strftime("%d %b %Y %H:%M IST", time.gmtime(ts + 19800))


def _sitrep_data(st):
    now = _now()
    areas = {}
    for i in st["incidents"]:
        a = areas.setdefault(i["zone"], {"zone": i["zone"], "waiting": [], "on_trucks": 0, "delivered_units": 0})
        if i["qty"] > 0 and not i["closed"]:
            a["waiting"].append({"item": i["item"], "qty": i["qty"], "urgency": i["urgency"],
                                 "waiting_min": (now - i["reported_at"]) // 60, "note": i["note"]})
        a["on_trucks"] += i["assigned"]
        a["delivered_units"] += i["delivered"]
    active = [{"route": r["id"], "truck": r["truck_id"], "minutes_out": (now - r["created_at"]) // 60,
               "stops_left": sum(s["status"] == "pending" for s in r["stops"])}
              for r in st["routes"] if r["status"] == "active"]
    return {"time": _ist(now), "operation_started": _ist(st["started"]), "figures": summary(st),
            "areas": list(areas.values()), "hazards": [h["label"] for h in _active_hazards(st)],
            "active_routes": active, "stock": {d["name"]: d["stock"] for d in st["depots"]},
            "recent_waves": st["waves"][-3:]}


SITREP_PROMPT = """You are the duty officer of a Chennai flood-relief operation. Write a situation report for the
coordinator using ONLY the data below. Be specific (areas, items, quantities, minutes) and brief.
Return ONLY JSON: {"headline": "<one line>", "overview": "<2-3 sentences>",
"priorities": ["<most urgent unmet needs, max 5>"], "blocked_roads": ["..."],
"actions": ["<concrete next steps, max 5>"]}
Data: """


def _rule_sitrep(data):
    f = data["figures"]
    waiting = sorted(((a["zone"], w) for a in data["areas"] for w in a["waiting"]),
                     key=lambda x: (-x[1]["urgency"], -x[1]["waiting_min"]))
    low = [(depot, item, qty) for depot, items in data["stock"].items() for item, qty in items.items() if qty < LOW_STOCK]
    actions = []
    if waiting:
        actions.append(f"Dispatch a wave for the {len(waiting)} waiting request(s), most urgent first."
                       + ("" if f["trucks_free"] else " No truck is free: recall a route or add a truck."))
    late = [r for r in data["active_routes"] if r["minutes_out"] > 120]
    actions += [f"Check on {r['truck']} ({r['route']}): out {r['minutes_out']} min with {r['stops_left']} stop(s) left."
                for r in late[:2]]
    actions += [f"Restock {item} at {depot}." for depot, item, _ in low[:2]]
    if data["hazards"]:
        actions.append(f"Verify reported hazards and clear any that reopened: {', '.join(data['hazards'][:3])}.")
    return {"headline": f"{f['waiting']} request(s) waiting, {f['en_route']} on trucks, {f['units_delivered']} units delivered",
            "overview": (f"Operation running since {data['operation_started']} with {f['waves']} dispatch wave(s). "
                         f"{f['trucks_out']} truck(s) out, {f['trucks_free']} free. Longest current wait is "
                         f"{f['longest_wait_min']} min. {f['hazards']} active road hazard(s)."),
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
    lines += ["", f"Figures: {f['waiting']} waiting, {f['en_route']} on trucks, {f['delivered']} delivered, "
                  f"{f['units_delivered']} units delivered, {f['hazards']} active hazards, {f['waves']} waves, "
                  f"{f['trucks_out']} trucks out.",
              f"Generated by {'Gemini' if generated_by == 'gemini' else 'rule-based fallback'} from live operations data."]
    return "\n".join(lines)


def sitrep():
    heals = []
    data = _sitrep_data(load())
    fallback, rep, by = _rule_sitrep(data), None, "rules"
    if agent.llm_available():
        try:
            # stock thresholds are exact facts, not judgement: they always come from the rule, and Gemini
            # does not see raw stock (it tended to list healthy depots as alerts)
            facts = {k: v for k, v in data.items() if k != "stock"}
            rep, by = _clean(agent.gemini_json([SITREP_PROMPT + json.dumps(facts)], heals), fallback), "gemini"
            rep["stock_alerts"] = fallback["stock_alerts"]
        except Exception as e:
            heals.append(f"Situation report via Gemini failed ({str(e)[:60]}) -> rule-based report")
    rep = rep or fallback
    return {"generated_by": by, "report": rep, "text": render(rep, data, by), "heals": heals, "data": data}
