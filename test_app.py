"""Offline checks: simulator contract, self-healing, learning loop, API end to end."""
import os

os.environ["OFFLINE"] = "1"

import pytest
from fastapi.testclient import TestClient

import agent
from env import ReliefEnv


@pytest.fixture(autouse=True)
def tmp_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("OPS_PATH", str(tmp_path / "ops.json"))


def test_env_contract_reset_step_state():
    env = ReliefEnv()
    obs = env.reset(seed=1)
    assert {"depots", "pending_needs", "known_hazards", "route_matrix"} <= obs.keys()
    res = env.step({"assignments": agent.heuristic_plan(obs, "urgency_first")})
    assert {"observation", "reward", "done", "info"} == res.keys()
    assert env.state()["step"] == 1


def test_invalid_actions_are_ignored_not_crashing():
    env = ReliefEnv()
    env.reset(seed=2, chaos=False)
    res = env.step({"assignments": [{"need_id": "nope", "depot_id": "D9"}, "garbage"]})
    assert res["info"]["invalid"] == 2


def test_chaos_blocks_a_route_and_agent_heals():
    r = agent.run_episode(seed=3, chaos=True, strategy="nearest_first")
    first = r["trace"][0]["info"]
    assert first["new_hazards"], "chaos should spawn a hazard"
    assert first["blocked"], "the longest route should be blocked"
    healed = {b["need_id"] for b in first["blocked"]}
    delivered_later = {d["need_id"] for t in r["trace"][1:] for d in t["info"]["delivered"]}
    assert healed & delivered_later, "blocked deliveries must be replanned and delivered"
    assert any("replanning" in h for h in r["heals"])


def test_guardrail_repairs_bad_llm_plan():
    env = ReliefEnv()
    obs = env.reset(seed=4)
    heals = []
    fixed = agent.repair(obs, [{"need_id": "BAD", "depot_id": "D1"}], heals)
    assert {a["need_id"] for a in fixed} <= {n["id"] for n in obs["pending_needs"]}
    assert len(fixed) > 0 and heals


def test_learning_updates_memory_and_bandit_prefers_better_policy():
    for _ in range(12):
        agent.run_episode(chaos=True)
    mem = agent.load_memory()
    assert len(mem["episodes"]) == 12 and mem["lessons"]
    tried = {k: v for k, v in mem["bandit"].items() if v["n"]}
    assert len(tried) >= 3
    assert agent.select_strategy(mem, explore=False) != "fifo"


def test_score_is_percentage_of_best_case():
    for seed in range(40):  # includes split deliveries across steps
        for strategy in ("urgency_first", "nearest_first", "fifo"):
            r = agent.run_episode(seed=seed, chaos=True, strategy=strategy)
            assert 0 <= r["score"] <= 100, (seed, strategy, r["score"])
    r = agent.run_episode(seed=5, chaos=False, strategy="urgency_first")
    assert agent.load_memory()["episodes"][-1]["score"] == r["score"]


def test_memory_restored_from_browser_copy_after_cold_start():
    for _ in range(3):
        agent.run_episode(chaos=True)
    browser_copy = agent.load_memory()
    agent.reset_memory()  # simulate a serverless instance losing /tmp
    agent.sync_memory(browser_copy)
    assert len(agent.load_memory()["episodes"]) == 3
    agent.sync_memory({"episodes": "garbage"})  # malformed input is ignored
    agent.sync_memory({"episodes": [{}], "bandit": {"evil": 1}})  # behind server copy: ignored
    assert len(agent.load_memory()["episodes"]) == 3


def test_mcp_protocol_and_tools():
    from main import app

    c = TestClient(app)
    rpc = lambda method, params=None, mid=1: c.post("/mcp", json={"jsonrpc": "2.0", "id": mid, "method": method,
                                                                   "params": params or {}})
    init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "t", "version": "0"}}).json()["result"]
    assert init["protocolVersion"] == "2025-06-18" and "tools" in init["capabilities"]
    assert c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
    names = {t["name"] for t in rpc("tools/list").json()["result"]["tools"]}
    assert names == {"parse_field_report", "dispatch_relief", "train_agent", "get_memory"}
    out = rpc("tools/call", {"name": "dispatch_relief", "arguments": {"text": "Urgent 90 water in Adyar"}}).json()
    assert out["result"]["isError"] is False and out["result"]["structuredContent"]["deliveries"]
    assert rpc("tools/call", {"name": "nope"}).json()["error"]["code"] == -32602
    assert rpc("bogus").json()["error"]["code"] == -32601


def test_quota_circuit_breaker_skips_exhausted_model(monkeypatch):
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append(model)
            if model == "big":
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota")
            return type("R", (), {"text": '{"ok": true}'})()

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    import google.genai

    monkeypatch.setattr(google.genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_MODELS", "big,lite")
    monkeypatch.setattr(agent, "_cooldown", {})
    heals = []
    assert agent.gemini_json(["x"], heals) == {"ok": True}
    assert calls == ["big", "lite"] and "circuit breaker" in heals[0]  # no pointless retry on a spent quota
    heals2 = []
    assert agent.gemini_json(["x"], heals2) == {"ok": True}
    assert calls[2:] == ["lite"] and heals2 == []  # breaker open: exhausted model skipped entirely


def _calm_env(seed=11):
    """Deterministic scenario: no floods, plenty of stock, so only truck rules matter."""
    import env as envmod

    e = ReliefEnv()
    e.reset(seed=seed, chaos=False)
    e._s["hazards"] = []
    for d in e._s["depots"]:
        d["stock"] = {it: 5000 for it in envmod.ITEMS}
    return e, envmod


def test_truck_capacity_caps_a_single_delivery():
    e, envmod = _calm_env()
    n = e._s["needs"][0]
    n["qty"] = n["requested"] = 600
    res = e.step({"assignments": [{"need_id": n["id"], "depot_id": "D1"}]})
    assert res["info"]["delivered"][0]["qty"] == envmod.TRUCK_CAPACITY
    assert n["id"] in [m["id"] for m in res["observation"]["pending_needs"]]  # rest waits for the next round


def test_multi_stop_route_is_driven_in_order():
    e, envmod = _calm_env()
    for n in e._s["needs"]:
        n["qty"] = n["requested"] = 40
    ids = [n["id"] for n in e._s["needs"][:3]]
    res = e.step({"assignments": [{"need_id": i, "depot_id": "D1", "truck_id": "D1-T1"} for i in ids]})
    legs = res["info"]["delivered"]
    assert [d["stop"] for d in legs] == [1, 2, 3] and {d["truck_id"] for d in legs} == {"D1-T1"}
    assert legs[0]["minutes"] <= legs[1]["minutes"] <= legs[2]["minutes"]  # later stops arrive later
    first = e._need(ids[0])
    assert (legs[1]["from"]["lat"], legs[1]["from"]["lng"]) == (first["lat"], first["lng"])


def test_no_truck_when_depot_fleet_is_full():
    e, envmod = _calm_env()
    for n in e._s["needs"]:
        n["qty"] = n["requested"] = 250
    res = e.step({"assignments": [{"need_id": n["id"], "depot_id": "D2"} for n in e._s["needs"][:3]]})
    assert len(res["info"]["delivered"]) == envmod.TRUCKS_PER_DEPOT
    assert len(res["info"]["no_truck"]) == 1


def test_policy_decide_serves_every_stocked_need():
    import random as rnd

    import policy

    e = ReliefEnv()
    obs = e.reset(seed=21)
    r = rnd.Random(0)
    plan = policy.decide(obs, lambda f: r.random(), lambda f: r.random(), lambda logits, kind: r.randrange(len(logits)))
    stocked = {n["id"] for n in obs["pending_needs"]
               if any(d["stock"].get(n["item"], 0) > 0 for d in obs["depots"])}
    assert {a["need_id"] for a in plan} == stocked
    assert e.step({"assignments": plan})["info"]["invalid"] == 0


def test_trained_policy_runs_without_pytorch():
    import policy

    if not policy.available():
        pytest.skip("policy_weights.json not trained yet")
    r = agent.run_episode(seed=5, chaos=True, strategy="rl_policy")
    assert r["strategy"] == "rl_policy" and 0 <= r["score"] <= 100 and r["deliveries"]


def test_openenv_environment_and_server():
    pytest.importorskip("openenv")
    from relief_openenv import Assignment, ReliefAction
    from relief_openenv.server.app import app as openenv_app
    from relief_openenv.server.environment import ReliefEnvironment

    env = ReliefEnvironment()
    obs = env.reset(seed=3, chaos=True)
    assert not obs.done and obs.pending_needs and obs.truck_capacity
    while not obs.done:
        plan = agent.heuristic_plan(obs.model_dump(), "urgency_first")
        obs = env.step(ReliefAction(assignments=[Assignment(**a) for a in plan]))
    assert obs.score is not None and 0 <= obs.score <= 100 and env.state.step_count >= 1
    c = TestClient(openenv_app)
    assert c.get("/health").status_code == 200
    schema = c.get("/schema").json()
    assert "assignments" in str(schema)


def test_operations_queue_wave_and_sitrep():
    import main

    main._hits.clear()
    c = TestClient(main.app)
    assert "Field report" in c.get("/report").text
    r = c.post("/api/reports", data={"text": "Urgent: 120 people in Velachery need water and medicine",
                                     "source": "phone", "lat": "12.9790", "lng": "80.2200"}).json()
    assert [i["item"] for i in r["incidents"]] == ["water", "medicine"]
    assert all(i["located_by"] == "gps" and i["lat"] == 12.979 and i["status"] == "waiting" for i in r["incidents"])
    assert c.get("/api/ops").json()["summary"]["waiting"] == 2

    w = c.post("/api/ops/wave", json={"chaos": False}).json()
    units = sum(d["qty"] for d in w["deliveries"])
    assert w["wave"]["n"] == 1 and units > 0
    stock_left = sum(sum(items.values()) for items in w["ops"]["stock"].values())
    assert stock_left == 3 * 4 * 400 - units  # depots start with 400 of each item
    delivered = [i for i in w["ops"]["incidents"] if i["status"] == "delivered"]
    assert delivered and all(i["qty"] == 0 and "arrives" in i["note"] for i in delivered)

    s = c.post("/api/ops/sitrep").json()
    assert s["generated_by"] == "rules" and s["text"].startswith("SITUATION REPORT")
    assert "Recommended actions" in s["text"]

    assert c.post("/api/ops/reset").json()["summary"]["waiting"] == 0
    assert c.post("/api/ops/wave", json={}).status_code == 409  # nothing to dispatch
    assert c.post("/api/reports", data={"text": ""}).status_code == 400
    main._hits.clear()


def test_operation_keeps_collapsed_roads_and_restores_from_browser():
    import ops

    report = {"summary": "test", "needs": [agent._clean_need({"zone": "Ennore", "item": "food", "qty": 90, "urgency": 4}),
                                           agent._clean_need({"zone": "Porur", "item": "water", "qty": 60, "urgency": 5})],
              "hazards": []}
    ops.add_report(report)
    out = ops.run_wave(chaos=True)
    assert out["trace"][0]["info"]["new_hazards"]
    assert any(h["label"].startswith("Bridge collapsed") for h in out["ops"]["hazards"])  # stays for later waves
    browser_copy = out["ops"]
    ops.reset()  # a restarted serverless instance
    ops.sync(browser_copy)
    restored = ops.load()
    assert restored["version"] == browser_copy["version"] and len(restored["incidents"]) == 2
    ops.sync({"version": 999, "incidents": [{"id": "I1", "item": "gold", "lat": 0, "lng": 0}]})  # junk is filtered
    assert all(i["item"] in ("food", "water") for i in ops.load()["incidents"])


def test_rate_limits_per_client(monkeypatch):
    import main

    monkeypatch.setattr(main, "LIMITS", {**main.LIMITS, "dispatch": (2, 600), "mcp": (1, 600)})
    main._hits.clear()
    c = TestClient(main.app)
    for _ in range(2):
        assert c.post("/api/dispatch", data={"text": "50 food in Adyar"}).status_code == 200
    r = c.post("/api/dispatch", data={"text": "50 food in Adyar"})
    assert r.status_code == 429 and "Retry-After" in r.headers
    other = c.post("/api/dispatch", data={"text": "50 food"}, headers={"x-forwarded-for": "203.0.113.9"})
    assert other.status_code == 200  # limits are per client address
    call = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "dispatch_relief",
                                                                          "arguments": {"text": "x"}}}
    assert c.post("/mcp", json=call).status_code == 200
    blocked = c.post("/mcp", json=call)
    assert blocked.status_code == 429 and blocked.json()["error"]["code"] == -32000
    assert c.post("/mcp", json={"jsonrpc": "2.0", "id": 8, "method": "tools/list"}).status_code == 200  # cheap: never limited
    main._hits.clear()


def test_daily_gemini_budget_degrades_to_rules(monkeypatch):
    monkeypatch.setenv("GEMINI_DAILY_BUDGET", "1")
    monkeypatch.setattr(agent, "_budget", {"day": "", "used": 0})
    agent._spend_budget()
    with pytest.raises(RuntimeError, match="budget"):
        agent._spend_budget()
    assert agent.budget_status() == {"used_today": 1, "daily_limit": 1}


def test_offline_intake_extracts_need():
    out = agent.intake("Urgent: 120 people in Velachery need water")
    n = out["needs"][0]
    assert (n["zone"], n["item"], n["qty"], n["urgency"]) == ("Velachery", "water", 120, 5)
    camp = agent.intake("Royapuram camp: 150 people need food packets and blankets. Ennore road is underwater.")
    assert [n["item"] for n in camp["needs"]] == ["food", "blankets"]  # "underwater" is not a water request
    tanglish = agent.intake("Velachery la 120 families, kudika thanni and medicine urgent-a venum")
    assert [n["item"] for n in tanglish["needs"]] == ["water", "medicine"]


def test_api_end_to_end():
    from main import app

    c = TestClient(app)
    assert c.get("/healthz").json()["ok"]
    assert "ReliefRoute" in c.get("/").text
    r = c.post("/api/dispatch", data={"text": "50 blankets needed in Adyar", "chaos": "true"},
               files={"image": ("x.png", b"\x89PNG fake", "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["intake"]["needs"] and body["trace"] and any(n["source"] == "field" for n in body["initial"]["needs"])
    t = c.post("/api/train", json={"episodes": 3}).json()
    assert len(t["episodes"]) == 3 and t["last"]["trace"]
    s = c.get("/api/scenario").json()
    assert s["initial"]["depots"] and s["initial"]["needs"]
    assert c.post("/api/memory/reset").json()["episodes"] == []
