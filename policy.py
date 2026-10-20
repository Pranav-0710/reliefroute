"""Learned dispatch policy: features, decision procedure and pure-Python inference.

Two small networks score the two decisions a dispatcher makes in each round:
    need net:  which pending request to load next (this also sets each truck's stop order)
    depot net: which depot (and therefore which of its trucks) serves it
While deciding, the policy simulates the depot fleets exactly as the simulator loads them, so its
features see real arrival times, the actual leg each truck would drive (previous stop -> next stop)
and the space left on the truck. Trained with REINFORCE in the flood simulator
(rl/train_policy.py, PyTorch) and exported to policy_weights.json, so the app needs no PyTorch.
"""
import json
import math
from pathlib import Path

from env import DETOUR_FACTOR, SPEED_KMH, _segment_hits, distance_km

WEIGHTS_PATH = Path(__file__).parent / "policy_weights.json"
N_NEED, N_PAIR = 8, 11


def _fleet(obs):
    """Mirror of env.ReliefEnv._new_fleet: every truck starts the round empty at its depot."""
    return {t: {"depot": d["id"], "pos": (d["lat"], d["lng"]), "cap": obs["truck_capacity"], "hours": 0.0, "stops": 0}
            for d in obs["depots"] for t in d["trucks"]}


def _truck_for(fleet, depot_id):
    """Mirror of env.ReliefEnv._pick_truck without a requested truck: the earliest-free one."""
    free = [t for t, v in fleet.items() if v["depot"] == depot_id and v["cap"] > 0]
    return min(free, key=lambda t: (fleet[t]["hours"], fleet[t]["stops"], t)) if free else None


def _leg(obs, truck, n):
    """km of the leg this truck would drive to n (detoured if it crosses a known hazard), and whether it does."""
    to = (n["lat"], n["lng"])
    blocked = any(_segment_hits(truck["pos"], to, h) for h in obs["known_hazards"])
    return distance_km(truck["pos"], to) * (DETOUR_FACTOR if blocked else 1), blocked


def _options(n, obs, stock, fleet):
    """Depots that can serve n now: (depot_id, truck_id, leg_km, leg_blocked, arrival_hours)."""
    out = []
    for d in stock:
        if stock[d].get(n["item"], 0) <= 0:
            continue
        t = _truck_for(fleet, d)
        if t is None:
            continue
        km, blocked = _leg(obs, fleet[t], n)
        out.append((d, t, km, blocked, fleet[t]["hours"] + km / SPEED_KMH))
    return out


def need_features(n, obs, stock, fleet):
    opts = _options(n, obs, stock, fleet)
    demand = sum(m["qty"] for m in obs["pending_needs"] if m["item"] == n["item"]) or 1
    supply = sum(s.get(n["item"], 0) for s in stock.values())
    return [n["urgency"] / 5, min(n["qty"] / 150, 2.0),
            min((min(o[4] for o in opts) if opts else 3.0) / 2, 1.5),
            1.0 if any(stock[d].get(n["item"], 0) >= n["qty"] for d in stock) else 0.0,
            min(supply / demand, 3) / 3,
            1.0 if any(fleet[o[1]]["cap"] >= n["qty"] for o in opts) else 0.0,
            obs["step"] / max(obs["max_steps"], 1), 1.0]


def pair_features(n, opt, opts, obs, stock, fleet):
    d, t, km, blocked, arrive = opt
    have = stock[d].get(n["item"], 0)
    demand = sum(m["qty"] for m in obs["pending_needs"] if m["item"] == n["item"]) or 1
    return [min(arrive / 2, 1.5), 1.0 if blocked else 0.0, min(km / 30, 1.5),
            min(have / n["qty"], 1.0), 1.0 if have >= n["qty"] else 0.0,
            min(fleet[t]["cap"] / n["qty"], 1.0),
            1.0 if arrive <= min(o[4] for o in opts) + 1e-9 else 0.0,
            min(fleet[t]["stops"] / 3, 1.0), min(have / demand, 3) / 3, n["urgency"] / 5, 1.0]


def decide(obs, score_need, score_pair, choose):
    """Build one round of assignments.

    score_need / score_pair map a feature list to a logit; choose(logits, kind) returns an index
    (argmax at inference, a sample during training). Framework-agnostic so training and serving
    share exactly the same procedure.
    """
    stock = {d["id"]: dict(d["stock"]) for d in obs["depots"]}
    fleet = _fleet(obs)
    remaining = list(obs["pending_needs"])
    out = []
    while remaining:
        n = remaining.pop(choose([score_need(need_features(m, obs, stock, fleet)) for m in remaining], "need"))
        opts = _options(n, obs, stock, fleet)
        if not opts:
            continue
        opt = opts[choose([score_pair(pair_features(n, o, opts, obs, stock, fleet)) for o in opts], "depot")]
        d, t, km, blocked, arrive = opt
        load = min(n["qty"], stock[d][n["item"]], fleet[t]["cap"])
        stock[d][n["item"]] -= load
        fleet[t].update(pos=(n["lat"], n["lng"]), cap=fleet[t]["cap"] - load, hours=arrive, stops=fleet[t]["stops"] + 1)
        out.append({"need_id": n["id"], "depot_id": d, "truck_id": t, "detour": blocked})
    return out


# ---- inference without PyTorch ----------------------------------------------------
_weights = None


def load():
    global _weights
    if _weights is None and WEIGHTS_PATH.exists():
        w = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        if len(w["need"]["W1"][0]) == N_NEED and len(w["depot"]["W1"][0]) == N_PAIR:  # ignore stale weights
            _weights = w
    return _weights


def available():
    return load() is not None


def mlp(w, x):
    """tanh MLP with one hidden layer, same shape as rl/train_policy.Scorer."""
    h = [math.tanh(sum(a * b for a, b in zip(row, x)) + bias) for row, bias in zip(w["W1"], w["b1"])]
    return sum(a * b for a, b in zip(w["W2"], h)) + w["b2"]


def plan(obs):
    w = load()
    argmax = lambda logits, kind: max(range(len(logits)), key=logits.__getitem__)
    return decide(obs, lambda f: mlp(w["need"], f), lambda f: mlp(w["depot"], f), argmax)
