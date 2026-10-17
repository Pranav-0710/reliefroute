"""OpenEnv-style disaster-relief logistics simulator.

Same contract as PyTorch OpenEnv environments:
    reset() -> observation
    step(action) -> {"observation", "reward", "done", "info"}
    state() -> full episode state
Runs in-process (no Docker) so the agent can play many episodes fast; it can be
wrapped as an OpenEnv HTTP server later for RL training.
"""
import math
import random

CENTER = (13.05, 80.24)  # Chennai
ITEMS = ["water", "food", "medicine", "blankets"]
SPEED_KMH = 25
DETOUR_FACTOR = 1.6
MAX_STEPS = 3

DEPOTS = [
    ("D1", "Guindy Hub", 13.0067, 80.2206),
    ("D2", "Anna Nagar Store", 13.0850, 80.2101),
    ("D3", "Tondiarpet Warehouse", 13.1260, 80.2880),
]
ZONES = [
    ("Velachery", 12.9815, 80.2180),
    ("Saidapet", 13.0213, 80.2231),
    ("T Nagar", 13.0418, 80.2341),
    ("Mylapore", 13.0368, 80.2676),
    ("Perambur", 13.1180, 80.2330),
    ("Royapuram", 13.1137, 80.2954),
    ("Adyar", 13.0012, 80.2565),
    ("Kodambakkam", 13.0521, 80.2255),
    ("Ennore", 13.2146, 80.3203),
    ("Porur", 13.0382, 80.1565),
]


def _xy(lat, lng):
    """Local flat projection in km around the city centre."""
    return ((lng - CENTER[1]) * 111.32 * math.cos(math.radians(CENTER[0])),
            (lat - CENTER[0]) * 110.57)


def distance_km(a, b):
    (x1, y1), (x2, y2) = _xy(*a), _xy(*b)
    return math.hypot(x2 - x1, y2 - y1) * 1.3  # road winding factor


def _segment_hits(a, b, hazard):
    (ax, ay), (bx, by) = _xy(*a), _xy(*b)
    hx, hy = _xy(hazard["lat"], hazard["lng"])
    dx, dy = bx - ax, by - ay
    t = ((hx - ax) * dx + (hy - ay) * dy) / ((dx * dx + dy * dy) or 1e-9)
    t = max(0.0, min(1.0, t))
    return math.hypot(ax + t * dx - hx, ay + t * dy - hy) <= hazard["radius_km"]


def _pt(x):
    return (x["lat"], x["lng"])


class ReliefEnv:
    def __init__(self):
        self._s = None
        self._rng = random.Random()

    # ---- OpenEnv contract -------------------------------------------------
    def reset(self, seed=None, extra_needs=None, extra_hazards=None, chaos=True, n_needs=5):
        rng = self._rng = random.Random(seed)
        depots = [{"id": i, "name": n, "lat": la, "lng": lo,
                   "stock": {it: rng.randint(60, 260) for it in ITEMS}}
                  for i, n, la, lo in DEPOTS]
        needs = []
        for k in range(n_needs):
            zone, la, lo = rng.choice(ZONES)
            needs.append({"id": f"N{k + 1}", "zone": zone,
                          "lat": la + rng.uniform(-0.01, 0.01), "lng": lo + rng.uniform(-0.01, 0.01),
                          "item": rng.choice(ITEMS), "qty": rng.randint(30, 150),
                          "urgency": rng.randint(1, 5), "source": "sim"})
        for k, n in enumerate(extra_needs or []):
            needs.append({**n, "id": f"R{k + 1}", "source": "field"})
        hazards = []
        for k in range(rng.randint(1, 2)):
            zone, la, lo = rng.choice(ZONES)
            hazards.append({"id": f"H{k + 1}", "label": f"Flooded road near {zone}",
                            "lat": la + 0.012, "lng": lo + 0.012, "radius_km": 1.0, "known": True})
        for h in extra_hazards or []:
            hazards.append({"id": f"H{len(hazards) + 1}", "label": h.get("label", "Field-reported hazard"),
                            "lat": h["lat"], "lng": h["lng"], "radius_km": 0.8, "known": True})
        for n in needs:
            n["requested"] = n["qty"]
        self._s = {"seed": seed, "step": 0, "depots": depots, "needs": needs, "hazards": hazards,
                   "pending": [n["id"] for n in needs], "deliveries": [], "events": [],
                   "total_reward": 0.0, "done": False, "chaos": chaos,
                   # best case: every need fully served instantly; used to normalise scores to %
                   "max_reward": sum(n["urgency"] * 10 for n in needs)}
        return self.observation()

    def step(self, action):
        s = self._s
        if s is None or s["done"]:
            raise RuntimeError("episode is done; call reset()")
        s["step"] += 1
        raw = action.get("assignments") or []
        plan, seen = [], set()
        for a in raw:
            if isinstance(a, dict) and a.get("need_id") in s["pending"] and self._depot(a.get("depot_id")) \
                    and a["need_id"] not in seen:
                seen.add(a["need_id"])
                plan.append(a)
        info = {"delivered": [], "blocked": [], "stockout": [], "invalid": len(raw) - len(plan),
                "new_hazards": []}

        # Surprise event: a bridge collapses on the longest route that is not detouring.
        if s["chaos"] and s["step"] == 1:
            cands = [a for a in plan if not a.get("detour")]
            if cands:
                a = max(cands, key=lambda a: distance_km(_pt(self._depot(a["depot_id"])), _pt(self._need(a["need_id"]))))
                d, n = self._depot(a["depot_id"]), self._need(a["need_id"])
                h = {"id": f"H{len(s['hazards']) + 1}", "label": "Bridge collapsed (live report)",
                     "lat": (d["lat"] + n["lat"]) / 2, "lng": (d["lng"] + n["lng"]) / 2,
                     "radius_km": 0.6, "known": True}
                s["hazards"].append(h)
                info["new_hazards"].append(h)

        reward = 0.0
        for a in plan:
            n, d = self._need(a["need_id"]), self._depot(a["depot_id"])
            hits = self._blocked(d, n)
            if hits and not a.get("detour"):
                reward -= 5
                info["blocked"].append({"need_id": n["id"], "depot_id": d["id"], "hazards": hits})
                continue
            qty = min(n["qty"], d["stock"].get(n["item"], 0))
            if qty <= 0:
                reward -= 2
                info["stockout"].append({"need_id": n["id"], "depot_id": d["id"]})
                continue
            d["stock"][n["item"]] -= qty
            km = distance_km(_pt(d), _pt(n)) * (DETOUR_FACTOR if a.get("detour") else 1)
            hours = km / SPEED_KMH
            # credit relative to the original request, so split deliveries can't exceed the best case
            reward += n["urgency"] * 10 * qty / n["requested"] - n["urgency"] * hours * 4
            n["qty"] -= qty
            if n["qty"] <= 0:
                s["pending"].remove(n["id"])
            delivery = {"step": s["step"], "need_id": n["id"], "depot_id": d["id"], "item": n["item"],
                        "qty": qty, "km": round(km, 1), "minutes": round(hours * 60),
                        "detour": bool(a.get("detour"))}
            s["deliveries"].append(delivery)
            info["delivered"].append(delivery)

        done = not s["pending"] or s["step"] >= MAX_STEPS
        if done:
            for nid in s["pending"]:
                n = self._need(nid)
                reward -= n["urgency"] * 10 * n["qty"] / n["requested"]  # penalise only the undelivered share
            info["unmet"] = list(s["pending"])
        s["total_reward"] += reward
        s["done"] = done
        s["events"].append({"step": s["step"], "reward": round(reward, 1), "info": info})
        return {"observation": self.observation(), "reward": round(reward, 2), "done": done, "info": info}

    def state(self):
        s = self._s
        return {**s, "total_reward": round(s["total_reward"], 2)}

    # ---- helpers ----------------------------------------------------------
    def observation(self):
        s = self._s
        pend = [self._need(i) for i in s["pending"]]
        return {
            "step": s["step"], "max_steps": MAX_STEPS,
            "depots": [{"id": d["id"], "name": d["name"], "lat": d["lat"], "lng": d["lng"],
                        "stock": dict(d["stock"])} for d in s["depots"]],
            "pending_needs": [dict(n) for n in pend],
            "known_hazards": [dict(h) for h in s["hazards"] if h["known"]],
            "route_matrix": {n["id"]: {d["id"]: {"km": round(distance_km(_pt(d), _pt(n)), 1),
                                                 "blocked_by": self._blocked(d, n)}
                                       for d in s["depots"]} for n in pend},
        }

    def _blocked(self, depot, need):
        return [h["id"] for h in self._s["hazards"] if _segment_hits(_pt(depot), _pt(need), h)]

    def _need(self, nid):
        return next((n for n in self._s["needs"] if n["id"] == nid), None)

    def _depot(self, did):
        return next((d for d in self._s["depots"] if d["id"] == did), None)
