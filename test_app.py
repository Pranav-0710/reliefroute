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
    assert names == {"parse_field_report", "dispatch_relief", "train_agent", "get_operation", "get_memory"}
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


def _need(zone, item, qty, urgency=3):
    return agent._clean_need({"zone": zone, "item": item, "qty": qty, "urgency": urgency})


def _tokens():
    import ops

    return {t["id"]: t["token"] for t in ops.load()["trucks"]}


def test_operations_report_wave_driver_and_sitrep_over_the_api():
    import main
    import ops

    main._hits.clear()
    c = TestClient(main.app)
    assert "Field report" in c.get("/report").text
    r = c.post("/api/reports", data={"text": "Urgent: 120 people in Velachery need water and medicine",
                                     "source": "phone", "lat": "12.9790", "lng": "80.2200"}).json()
    assert [i["item"] for i in r["incidents"]] == ["water", "medicine"]
    assert all(i["located_by"] == "gps" and i["lat"] == 12.979 and i["status"] == "waiting" for i in r["incidents"])
    assert r["track_url"] == f"/track/{r['track']}" and "token" not in str(r["ops"]["reports"])
    assert c.get(r["track_url"]).status_code == 200
    assert [i["status"] for i in c.get(f"/api/track/{r['track']}").json()["incidents"]] == ["waiting", "waiting"]

    w = c.post("/api/ops/wave").json()
    units = sum(s["qty"] for route in w["routes"] for s in route["stops"])
    assert w["wave"]["n"] == 1 and units == 240 and w["summary"]["trucks_out"] == len(w["routes"])
    stock_left = sum(sum(d["stock"].values()) for d in w["ops"]["depots"])
    assert stock_left == 3 * 4 * 400 - units  # loaded onto the trucks at dispatch
    assert all(i["status"] == "en_route" for i in w["ops"]["incidents"])
    assert c.post("/api/reports", data={"text": "40 blankets needed in Ennore"}).status_code == 200

    route = w["routes"][0]
    tok = _tokens()[route["truck_id"]]
    assert c.get(f"/driver/{tok}").status_code == 200
    dv = c.get(f"/api/driver/{tok}").json()
    assert dv["route"]["id"] == route["id"] and dv["route"]["shape"]
    for s in route["stops"]:
        out = c.post(f"/api/driver/{tok}/stops/{s['n']}", json={"outcome": "delivered"}).json()
    assert out["closed"] and out["learned"]["score"] == 100.0 and out["route"] is None
    assert c.post(f"/api/driver/{tok}/stops/1", json={"outcome": "delivered"}).status_code == 409
    assert c.get("/api/driver/not-a-real-token").status_code == 404
    assert c.post(f"/api/driver/{tok}/stops/1", json={"outcome": "teleported"}).status_code == 409

    s = c.post("/api/ops/sitrep").json()
    assert s["generated_by"] == "rules" and s["text"].startswith("SITUATION REPORT")
    assert "Recommended actions" in s["text"]
    assert c.post("/api/ops/reset").json()["summary"]["waiting"] == 0
    assert c.post("/api/ops/wave").status_code == 409  # nothing to dispatch
    assert c.post("/api/reports", data={"text": ""}).status_code == 400
    assert ops.load()["log"][-1]["text"] == "Started a new operation"
    main._hits.clear()


def test_duplicate_reports_merge_instead_of_sending_two_trucks():
    import ops

    first = ops.add_report({"summary": "a", "hazards": [], "needs": [_need("Velachery", "water", 120, 3)]})
    again = ops.add_report({"summary": "b", "hazards": [], "needs": [_need("Velachery", "water", 150, 5),
                                                                    _need("Velachery", "food", 60)]})
    inc = first["incidents"][0]
    assert [i["id"] for i in again["merged"]] == [inc["id"]] and [i["item"] for i in again["incidents"]] == ["food"]
    merged = next(i for i in ops.load()["incidents"] if i["id"] == inc["id"])
    assert (merged["requested"], merged["qty"], merged["urgency"], merged["reports"]) == (150, 150, 5, 2)
    far = ops.add_report({"summary": "c", "hazards": [], "needs": [_need("Ennore", "water", 50)]})
    assert far["incidents"] and not far["merged"]  # another place is another need
    # a position guessed from "Velachery" is only locality-precise: a phone's GPS 1.2 km away is the same need
    phone = ops.add_report({"summary": "d", "hazards": [], "needs": [_need("Velachery", "water", 90)]},
                           source="phone", coords=(12.9815 + 0.011, 80.2180))
    assert [i["id"] for i in phone["merged"]] == [inc["id"]]
    other = ops.add_report({"summary": "e", "hazards": [], "needs": [_need("Velachery", "water", 30)]},
                           source="phone", coords=(12.9815 + 0.011 + 0.009, 80.2180))
    assert other["incidents"] and not other["merged"]  # two GPS positions 1 km apart are two needs
    assert [i["id"] for i in ops.track_view(again["track"])["incidents"]] == [again["incidents"][0]["id"], inc["id"]]


def test_failed_and_blocked_stops_return_to_queue_and_teach_the_agent(monkeypatch):
    import ops

    clock = [1_000_000]
    monkeypatch.setattr(ops, "_now", lambda: clock[0])
    ops.add_report({"summary": "t", "hazards": [], "needs": [_need("Adyar", "food", 60, 5), _need("Porur", "water", 40, 2),
                                                            _need("Royapuram", "medicine", 30, 4)]})
    w = ops.run_wave()
    stops = [(r, s) for r in w["routes"] for s in r["stops"]]
    assert len(stops) == 3
    tok = _tokens()
    stock_before = {d["id"]: dict(d["stock"]) for d in ops.load()["depots"]}
    (r1, s1), (r2, s2), (r3, s3) = stops
    clock[0] += int(s1["eta_raw"] * 60 * 2)  # the first truck arrives at twice the router's estimate
    a = ops.driver_update(tok[r1["truck_id"]], s1["n"], "delivered", qty=s1["qty"] - 10)
    b = ops.driver_update(tok[r2["truck_id"]], s2["n"], "blocked", note="bridge under water", coords=(13.03, 80.17))
    c = ops.driver_update(tok[r3["truck_id"]], s3["n"], "failed", note="camp moved")
    st = ops.load()
    incs = {i["id"]: i for i in st["incidents"]}
    assert incs[s1["incident_id"]]["delivered"] == s1["qty"] - 10 and incs[s1["incident_id"]]["qty"] == 10
    assert incs[s2["incident_id"]]["status"] == "waiting" and "road blocked" in incs[s2["incident_id"]]["note"]
    assert incs[s3["incident_id"]]["status"] == "waiting" and "camp moved" in incs[s3["incident_id"]]["note"]
    h = st["hazards"][-1]
    assert h["active"] and (h["lat"], h["lng"]) == (13.03, 80.17) and h["source"] == "driver"
    assert all(t["status"] == "idle" for t in st["trucks"])  # every route closed
    returned = {d["id"]: {k: d["stock"][k] - stock_before[d["id"]][k] for k in d["stock"]} for d in st["depots"]}
    assert returned[r1["depot_id"]][s1["item"]] >= 10  # undelivered cargo is back in stock
    assert a["learned"]["eta_factor"] > 1.0 and a["learned"]["eta_samples"] == 1  # it learned trucks run slow
    assert b["learned"]["score"] == 0 and c["learned"]["lessons"]
    mem = agent.load_memory()
    assert any("blocked" in l for l in mem["lessons"]) and any("failed" in l for l in mem["lessons"])
    w2 = ops.run_wave()
    assert w2["routes"][0]["eta_factor"] == mem["eta_factor"]  # new ETAs use the learned correction


def test_recall_resources_and_their_guards():
    import main

    main._hits.clear()
    c = TestClient(main.app)
    c.post("/api/reports", data={"text": "90 people need food in Adyar"})
    w = c.post("/api/ops/wave").json()
    route = w["routes"][0]
    assert c.patch(f"/api/ops/trucks/{route['truck_id']}", json={"in_service": False}).status_code == 409
    rec = c.post(f"/api/ops/routes/{route['id']}/recall").json()
    assert rec["learned"]["score"] == 0 and rec["summary"]["waiting"] == 1 and rec["summary"]["trucks_out"] == 0
    assert c.post(f"/api/ops/routes/{route['id']}/recall").status_code == 409

    d = c.post("/api/ops/depots", json={"name": "Velachery School", "lat": 12.98, "lng": 80.22}).json()
    depot = d["ops"]["depots"][-1]
    assert depot["id"] == "D4" and sum(depot["stock"].values()) == 0
    assert c.post("/api/ops/depots", json={"name": "Mumbai", "lat": 19.07, "lng": 72.87}).status_code == 409
    t = c.post("/api/ops/trucks", json={"depot_id": "D4", "capacity": 600}).json()
    truck = t["ops"]["trucks"][-1]
    assert (truck["id"], truck["capacity"], truck["status"]) == ("D4-T1", 600, "idle") and truck["token"]
    assert c.post("/api/ops/trucks", json={"depot_id": "D4", "capacity": 5}).status_code == 409
    rs = c.post("/api/ops/restock", json={"depot_id": "D4", "item": "food", "qty": 500, "note": "NGO convoy"}).json()
    assert rs["ops"]["depots"][-1]["stock"]["food"] == 500 and rs["ops"]["restocks"][-1]["note"] == "NGO convoy"
    assert c.post("/api/ops/restock", json={"depot_id": "D4", "item": "gold", "qty": 5}).status_code == 409
    off = c.patch("/api/ops/trucks/D1-T1", json={"in_service": False, "capacity": 300}).json()
    t1 = next(t for t in off["ops"]["trucks"] if t["id"] == "D1-T1")
    assert (t1["status"], t1["capacity"]) == ("out_of_service", 300)

    c.post("/api/reports", data={"text": "Road to Saidapet flooded, 30 water needed in Saidapet"})
    inc = next(i for i in c.get("/api/ops").json()["ops"]["incidents"] if i["zone"] == "Saidapet")
    closed = c.post(f"/api/ops/incidents/{inc['id']}/close", json={"reason": "resolved by local volunteers"}).json()
    assert next(i for i in closed["ops"]["incidents"] if i["id"] == inc["id"])["status"] == "closed"
    main._hits.clear()


def test_hazards_are_avoided_by_the_router_and_can_be_cleared(monkeypatch):
    import ops
    import roads

    sent = []

    class Resp:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def post(url, json, timeout):
        sent.append((url, json))
        if url.endswith("/sources_to_targets"):
            n = len(json["sources"])
            return Resp({"sources_to_targets": [[{"from_index": a, "to_index": b, "distance": 2.0 + a + b}
                                                 for b in range(n)] for a in range(n)]})
        legs = [{"summary": {"length": 3.5, "time": 600}, "shape": "_c`|@_c`|@_c`|@_c`|@"}
                for _ in json["locations"][1:]]
        return Resp({"trip": {"legs": legs}})

    monkeypatch.delenv("OFFLINE")
    monkeypatch.setattr(agent, "_gemini_key", lambda: None)
    monkeypatch.setattr(roads.httpx, "post", post)
    monkeypatch.setattr(roads, "_cache", type(roads._cache)())
    ops.add_report({"summary": "t", "needs": [_need("Adyar", "food", 60, 5)],
                    "hazards": [{"label": "Flooded underpass", "lat": 13.02, "lng": 80.245}]})
    w = ops.run_wave()
    r = w["routes"][0]
    assert w["wave"]["road"] and r["source"] == "road" and r["km"] == 3.5 and r["stops"][0]["eta_raw"] == 10
    assert r["shape"] == [[1.0, 1.0], [2.0, 2.0]]  # decoded polyline
    assert all("exclude_polygons" in body for _, body in sent)  # the hazard is an area to avoid
    h = ops.load()["hazards"][0]
    ops.clear_hazard(h["id"])
    assert not ops.load()["hazards"][0]["active"]

    def down(url, json, timeout):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(roads.httpx, "post", down)
    monkeypatch.setattr(roads, "_cache", type(roads._cache)())
    ops.add_report({"summary": "t", "hazards": [], "needs": [_need("Perambur", "water", 40, 4)]})
    w = ops.run_wave()
    assert w["routes"][0]["source"] == "estimate"  # self-healing: plans and routes on estimates
    assert any("Road router unavailable" in h for h in w["heals"]) and any("Road route for" in h for h in w["heals"])


def test_coordinator_sign_in_guards_every_change(monkeypatch):
    import auth
    import main

    main._hits.clear()
    monkeypatch.setenv("COORDINATOR_EMAILS", "lead@example.org")
    monkeypatch.setenv("FIREBASE_WEB_API_KEY", "web-key")
    monkeypatch.setenv("FIREBASE_PROJECT_ID", "relief-test")
    monkeypatch.setattr(auth, "_verified", {})
    users = {"good": {"email": "lead@example.org", "email_verified": True, "exp": 9e12},
             "stranger": {"email": "someone@example.org", "email_verified": True, "exp": 9e12}}

    def verify(token):
        if token not in users:
            raise ValueError("bad signature")
        return users[token]
    monkeypatch.setattr(auth, "_verify", verify)
    c = TestClient(main.app)
    good, stranger = {"Authorization": "Bearer good"}, {"Authorization": "Bearer stranger"}

    assert c.get("/api/config").json()["firebase"] == {"apiKey": "web-key", "projectId": "relief-test",
                                                         "authDomain": "relief-test.firebaseapp.com"}
    assert c.get("/healthz").json()["coordinator_sign_in"] is True
    snippet = """const firebaseConfig = {
  apiKey: "AIza-from-snippet",
  authDomain: "relief-test.firebaseapp.com",
};"""
    monkeypatch.setenv("FIREBASE_WEB_API_KEY", snippet)  # the whole snippet pasted from the Firebase console
    assert c.get("/api/config").json()["firebase"]["apiKey"] == "AIza-from-snippet"
    monkeypatch.setenv("FIREBASE_WEB_API_KEY", "web-key")
    assert c.post("/api/reports", data={"text": "50 food in Adyar"}).status_code == 200  # reporters never sign in
    for path in ("/api/ops/wave", "/api/ops/reset", "/api/ops/sitrep", "/api/memory/reset"):
        assert c.post(path).status_code == 401
        assert c.post(path, headers=stranger).status_code == 401
        assert c.post(path, headers={"Authorization": "Bearer forged"}).status_code == 401
    assert c.post("/api/ops/restock", json={"depot_id": "D1", "item": "food", "qty": 5}).status_code == 401
    assert all("token" not in t for t in c.get("/api/ops").json()["ops"]["trucks"])  # no driver links in public
    assert all(t["token"] for t in c.get("/api/ops", headers=good).json()["ops"]["trucks"])
    assert c.get("/api/me", headers=good).json() == {"coordinator": True, "email": "lead@example.org", "sign_in": True}
    w = c.post("/api/ops/wave", headers=good).json()
    assert w["wave"]["by"] == "lead@example.org" and w["ops"]["log"][-1]["by"] == "lead@example.org"
    main._hits.clear()


def test_sitrep_stock_alerts_are_never_left_to_gemini(monkeypatch):
    import ops

    ops.add_report({"summary": "t", "hazards": [],
                    "needs": [agent._clean_need({"zone": "Adyar", "item": "food", "qty": 50, "urgency": 3})]})
    monkeypatch.setattr(agent, "llm_available", lambda: True)
    monkeypatch.setattr(agent, "gemini_json", lambda parts, heals: {
        "headline": "h", "overview": "o", "priorities": ["p"], "blocked_roads": [], "actions": ["a"],
        "stock_alerts": ["Guindy Hub: water 400 (healthy stock wrongly flagged)"]})
    rep = ops.sitrep()
    assert rep["generated_by"] == "gemini" and rep["report"]["stock_alerts"] == []  # all depots full: no alerts
    assert "stock" not in ops.SITREP_PROMPT.split("Data:")[0].split("Return")[1]


def _fake_firestore(monkeypatch, interfere=0):
    """In-memory stand-in for the Firestore REST API, with the same write preconditions.
    `interfere` = number of writes that another 'instance' sneaks in just before ours."""
    import json as js

    import store

    docs, clock, left = {}, [0], [interfere]

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._body, self.text = code, body, js.dumps(body)

        def json(self):
            return self._body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.text)

    def get(url, headers, timeout):
        name = url.rsplit("/", 1)[1]
        if name not in docs:
            return Resp(404, {})
        body, stamp = docs[name]
        return Resp(200, {"fields": {"json": {"stringValue": body}}, "updateTime": stamp})

    def patch(url, params, timeout, headers, json):
        name = url.rsplit("/", 1)[1]
        if left[0] and name in docs:  # another instance writes between our read and our write
            left[0] -= 1
            other = js.loads(docs[name][0])
            other["written_by_other_instance"] = True
            clock[0] += 1
            docs[name] = (js.dumps(other), f"t{clock[0]}")
        cur = docs.get(name)
        if "currentDocument.updateTime" in params and (not cur or cur[1] != params["currentDocument.updateTime"]):
            return Resp(400, {"error": {"status": "FAILED_PRECONDITION"}})
        if params.get("currentDocument.exists") == "false" and cur:
            return Resp(409, {"error": {"status": "ALREADY_EXISTS"}})
        clock[0] += 1
        docs[name] = (json["fields"]["json"]["stringValue"], f"t{clock[0]}")
        return Resp(200, {})

    monkeypatch.setattr(store, "_service_account", lambda: {"project_id": "p", "client_email": "svc@p"})
    monkeypatch.setattr(store, "_token", lambda info: "token")
    monkeypatch.setattr(store.httpx, "get", get)
    monkeypatch.setattr(store.httpx, "patch", patch)
    return docs, left


def test_firestore_writes_never_lose_a_concurrent_update(monkeypatch):
    import store

    docs, left = _fake_firestore(monkeypatch)
    store.update("ops", "unused", dict, lambda st: st.update(a=1))
    left[0] = 1  # the next write races with another instance
    attempts = []
    store.update("ops", "unused", dict, lambda st: (attempts.append(1), st.update(b=2)))
    assert store.read("ops", "unused", dict) == {"a": 1, "written_by_other_instance": True, "b": 2}
    assert len(attempts) == 2  # first write rejected by the precondition, retried on fresh data
    assert store.backend() == "firestore"


def test_operations_and_memory_run_on_firestore(monkeypatch):
    import json as js

    import ops

    docs, _ = _fake_firestore(monkeypatch)
    ops.add_report({"summary": "t", "hazards": [],
                    "needs": [agent._clean_need({"zone": "Adyar", "item": "water", "qty": 70, "urgency": 5})]})
    out = ops.run_wave()
    assert out["wave"]["n"] == 1 and js.loads(docs["ops"][0])["waves"][0]["n"] == 1
    agent.run_episode(seed=1, strategy="urgency_first")
    assert len(js.loads(docs["memory"][0])["episodes"]) == 1  # memory lives in the shared document too


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
