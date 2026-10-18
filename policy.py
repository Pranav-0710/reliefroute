"""Learned dispatch policy: features, decision procedure and pure-Python inference.

Two small networks score the two decisions a dispatcher makes in each round:
    need net:  which pending request to serve next (this also sets the stop order)
    depot net: which depot serves it
They are trained with REINFORCE in the flood simulator (rl/train_policy.py, PyTorch)
and exported to policy_weights.json, so the deployed app runs them without PyTorch.
"""
import json
import math
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent / "policy_weights.json"
DETOUR = 1.6  # must match env.DETOUR_FACTOR


def _eff_km(obs, need_id, depot_id):
    o = obs["route_matrix"][need_id][depot_id]
    return o["km"] * (DETOUR if o["blocked_by"] else 1)


def need_features(n, obs, stock):
    have = [d for d in stock if stock[d].get(n["item"], 0) > 0]
    nearest = min((_eff_km(obs, n["id"], d) for d in have), default=45.0)
    demand = sum(m["qty"] for m in obs["pending_needs"] if m["item"] == n["item"]) or 1
    supply = sum(s.get(n["item"], 0) for s in stock.values())
    return [n["urgency"] / 5, n["qty"] / 150, nearest / 30,
            1.0 if any(stock[d].get(n["item"], 0) >= n["qty"] for d in stock) else 0.0,
            min(supply / demand, 3) / 3, 1.0]


def pair_features(n, depot_id, obs, stock):
    eff = _eff_km(obs, n["id"], depot_id)
    have = stock[depot_id].get(n["item"], 0)
    demand = sum(m["qty"] for m in obs["pending_needs"] if m["item"] == n["item"]) or 1
    effs = [_eff_km(obs, n["id"], d) for d in stock if stock[d].get(n["item"], 0) > 0]
    return [eff / 30, 1.0 if obs["route_matrix"][n["id"]][depot_id]["blocked_by"] else 0.0,
            min(have / n["qty"], 1.0), 1.0 if have >= n["qty"] else 0.0,
            1.0 if effs and eff <= min(effs) + 1e-9 else 0.0,
            min(have / demand, 3) / 3, n["urgency"] / 5, 1.0]


N_NEED, N_PAIR = 6, 8


def decide(obs, score_need, score_pair, choose):
    """Build one round of assignments.

    score_need / score_pair map a feature list to a logit; choose(logits, kind) returns an index
    (argmax at inference, a sample during training). Framework-agnostic so training and serving
    share exactly the same procedure.
    """
    stock = {d["id"]: dict(d["stock"]) for d in obs["depots"]}
    remaining = list(obs["pending_needs"])
    out = []
    while remaining:
        n = remaining.pop(choose([score_need(need_features(m, obs, stock)) for m in remaining], "need"))
        cands = [d for d in stock if stock[d].get(n["item"], 0) > 0]
        if not cands:
            continue
        did = cands[choose([score_pair(pair_features(n, d, obs, stock)) for d in cands], "depot")]
        stock[did][n["item"]] -= min(n["qty"], stock[did][n["item"]])
        out.append({"need_id": n["id"], "depot_id": did,
                    "detour": bool(obs["route_matrix"][n["id"]][did]["blocked_by"])})
    return out


# ---- inference without PyTorch ----------------------------------------------------
_weights = None


def load():
    global _weights
    if _weights is None and WEIGHTS_PATH.exists():
        _weights = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
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
