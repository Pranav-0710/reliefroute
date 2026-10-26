"""Autopilot: the agent decides when a dispatch wave is worth sending.

    off      waves only when a coordinator presses Dispatch
    propose  the agent plans a wave by itself and a coordinator approves or rejects it; a rejection
             with a reason becomes a lesson for later plans
    auto     the agent sends the wave itself (only after a coordinator approved at least one proposal)

When does it plan? An urgent request it has not planned for yet arrives, a request has waited longer
than max_wait_min, or something changed in the field (a truck came back, a stop failed, a road was cut;
ops sets autopilot.dirty). Never while a proposal is waiting, never twice within COOLDOWN_S, and never
for a request whose place is unknown.

Ticks come from the coordinator console while it is open, and optionally from an external scheduler
calling POST /api/agent/tick with AGENT_TICK_SECRET. Servers are stateless, so a lease in the operation
document (claimed with the same conditional write as everything else) keeps two instances from
planning at once.
"""
import agent
import ops

MODES = ("off", "propose", "auto")
LEASE_S = 90
COOLDOWN_S = 120
PROPOSAL_TTL_S = 600  # a plan older than this is stale: trucks and requests have moved on
KEEP_PROPOSALS = 20
AGENT = "agent (autopilot)"


def _eligible(st):
    return [i for i in st["incidents"] if i["qty"] > 0 and not i["closed"] and i["located_by"] != "guess"]


def _pending(st, now):
    return next((p for p in st["proposals"] if p["status"] == "pending" and p["expires_at"] > now), None)


def should_plan(st, now):
    """(plan now?, why). Reads only."""
    ap = st["autopilot"]
    if ap["mode"] == "off":
        return False, "autopilot is off"
    if _pending(st, now):
        return False, "a proposal is waiting for a coordinator"
    if ap["lease_until"] > now:
        return False, "another server is planning"
    if now - ap["last_plan_at"] < COOLDOWN_S:
        return False, "planned less than 2 minutes ago"
    waiting = _eligible(st)
    if not waiting:
        return False, "no request with a known place is waiting"
    if not any(t["status"] == "idle" for t in st["trucks"]):
        return False, "every truck is out"
    if ap["dirty"]:
        return True, ap["dirty"]
    seen = set(ap["considered"])
    urgent = [i for i in waiting if i["id"] not in seen and i["urgency"] >= ap["min_urgency"]]
    if urgent:
        return True, "urgent: " + ", ".join(f"{i['zone']} {i['item']}" for i in urgent[:3])
    oldest = max((now - i["reported_at"]) // 60 for i in waiting)
    if oldest >= ap["max_wait_min"] and now - ap["last_plan_at"] >= ap["max_wait_min"] * 60:
        return True, f"a request has waited {oldest} min"
    return False, f"nothing urgent; batching until a request waits {ap['max_wait_min']} min"


def _release(st, now, considered=None):
    ap = st["autopilot"]
    ap.update(lease_until=0, last_plan_at=now, dirty=None)
    if considered is not None:
        ap["considered"] = considered[-200:]


def _light(plans):
    """A decided proposal keeps what it was, not its road geometry (keeps the document small)."""
    return [{**p, "shape": []} for p in plans]


def _expire(st, now):
    for p in st["proposals"]:
        if p["status"] == "pending" and p["expires_at"] <= now:
            p.update(status="expired", plans=_light(p["plans"]))


def tick(now=None):
    """One autopilot step. Returns {"acted": bool, "why", "heals", and "proposal" or "wave"}."""
    now = now or ops._now()
    ok, why = should_plan(ops.load(), now)
    if not ok:  # the common case: no write at all
        return {"acted": False, "why": why, "heals": []}

    def claim(st):
        _expire(st, now)
        ok, why = should_plan(st, now)
        if not ok:
            raise ops.OpsError(why)
        st["autopilot"]["lease_until"] = now + LEASE_S
        return why, st["autopilot"]["mode"]
    try:
        why, mode = ops._update(claim)
    except ops.OpsError as e:
        return {"acted": False, "why": str(e), "heals": []}

    heals = []
    try:
        planned = ops.plan_wave(ops.load(), heals, skip_unlocated=True)
    except Exception as e:
        ops._update(lambda st: _release(st, now))
        if isinstance(e, ops.OpsError):
            return {"acted": False, "why": str(e), "heals": heals}
        raise
    considered = planned["needs"]
    if not planned["plans"]:
        def nothing(st):
            _release(st, now, considered)
            ops._log(st, AGENT, f"Planned for {len(considered)} request(s) ({why}) but no truck could take any "
                                f"stop (stock or capacity)")
        ops._update(nothing)
        return {"acted": False, "why": "no truck could take any stop", "heals": heals}

    if mode == "auto":
        def done(st, wave):
            _release(st, now, considered)
            wave["trigger"] = why
        out = ops.commit_wave(planned, AGENT, heals, hook=done)
        return {"acted": True, "why": why, "wave": out["wave"], "routes": out["routes"], "heals": out["heals"],
                "strategy_why": out["strategy_why"], "reasoning": out["reasoning"]}

    def propose(st):
        _release(st, now, considered)
        n = max([int(p["id"][1:]) for p in st["proposals"]] + [0]) + 1
        p = {"id": f"P{n}", "at": now, "expires_at": now + PROPOSAL_TTL_S, "trigger": why, "status": "pending",
             "strategy": planned["strategy"], "strategy_why": planned["strategy_why"], "reasoning": planned["reasoning"],
             "scores": planned["scores"], "eta_factor": planned["eta_factor"], "road": planned["road"],
             "needs": planned["needs"], "hazards": planned["hazards"], "info": planned["info"], "plans": planned["plans"],
             "routes": len(planned["plans"]), "units": sum(s["qty"] for pl in planned["plans"] for s in pl["stops"]),
             "by": None, "reason": "", "heals": heals[:10]}
        st["proposals"] = (st["proposals"] + [p])[-KEEP_PROPOSALS:]
        ops._log(st, AGENT, f"Proposed {p['id']} ({why}): {p['routes']} route(s), {p['units']} units, "
                            f"strategy {p['strategy']}")
        return p
    return {"acted": True, "why": why, "proposal": ops._update(propose), "heals": heals}


def approve(proposal_id, actor):
    """Send out the trucks of a pending proposal (re-validated: anything that changed meanwhile heals)."""
    st, now = ops.load(), ops._now()
    p = ops._find(st["proposals"], "id", proposal_id)
    if not p or p["status"] != "pending":
        raise ops.OpsError(f"{proposal_id} is not waiting for approval.")
    if p["expires_at"] <= now:
        ops._update(lambda st: _expire(st, now))
        raise ops.OpsError(f"{proposal_id} expired: trucks and requests have moved on. The agent will plan again.")
    planned = {k: p[k] for k in ("plans", "info", "strategy", "reasoning", "strategy_why", "scores", "eta_factor",
                                 "road", "needs", "hazards")}
    heals = list(p.get("heals", []))

    def decide(st, wave):
        live = ops._find(st["proposals"], "id", proposal_id)
        if not live or live["status"] != "pending":  # approved twice at once: only one wave goes out
            raise ops.OpsError(f"{proposal_id} was already decided.")
        live.update(status="approved", by=actor, decided_at=now, wave=wave["n"], plans=_light(live["plans"]))
        st["autopilot"]["approved"] += 1
        wave["trigger"] = f"{proposal_id} approved"
    return {**ops.commit_wave(planned, actor, heals, hook=decide), "proposal": proposal_id}


def reject(proposal_id, reason, actor):
    """Turn a proposal down. The reason becomes a lesson the LLM planner reads next time."""
    reason, now = str(reason or "").strip()[:120], ops._now()

    def upd(st):
        p = ops._find(st["proposals"], "id", proposal_id)
        if not p or p["status"] != "pending":
            raise ops.OpsError(f"{proposal_id} is not waiting for approval.")
        p.update(status="rejected", by=actor, decided_at=now, reason=reason, plans=_light(p["plans"]))
        st["autopilot"]["last_plan_at"] = now  # the same plan is not offered again straight away
        ops._log(st, actor, f"Rejected {proposal_id}" + (f": {reason}" if reason else ""))
        return p
    p = ops._update(upd)
    if reason:
        agent.add_lesson(f"Coordinator rejected a {p['strategy']} plan: {reason}")
    return p


def configure(mode=None, min_urgency=None, max_wait_min=None, actor="console"):
    def upd(st):
        ap = st["autopilot"]
        if mode is not None:
            if mode not in MODES:
                raise ops.OpsError(f"Autopilot mode must be one of {', '.join(MODES)}.")
            if mode == "auto" and not ap["approved"]:
                raise ops.OpsError("Approve at least one agent proposal before letting it dispatch on its own.")
            if mode != ap["mode"]:
                ops._log(st, actor, f"Autopilot {ap['mode']} -> {mode}")
            ap["mode"] = mode
            if mode == "off":
                for p in st["proposals"]:
                    if p["status"] == "pending":
                        p.update(status="withdrawn", plans=_light(p["plans"]))
        if min_urgency is not None:
            ap["min_urgency"] = max(1, min(5, int(min_urgency)))
        if max_wait_min is not None:
            ap["max_wait_min"] = max(5, min(240, int(max_wait_min)))
        return ap
    return ops._update(upd)
