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


def test_offline_intake_extracts_need():
    out = agent.intake("Urgent: 120 people in Velachery need water")
    n = out["needs"][0]
    assert (n["zone"], n["item"], n["qty"], n["urgency"]) == ("Velachery", "water", 120, 5)


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
