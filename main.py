"""ReliefRoute API: FastAPI server for the multimodal disaster-relief logistics agent."""
import json
import os
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import agent
import auth
import mcp_server
import ops
import roads

app = FastAPI(title="ReliefRoute")
STATIC = Path(__file__).parent / "static"

# ---- rate limits ----------------------------------------------------------------------
# Per client IP, sliding window, per server instance. Protects the shared Gemini quota and memory
# of a public deployment; not a substitute for authentication.
LIMITS = {"dispatch": (8, 600), "train": (60, 600), "mcp": (10, 600), "reset": (5, 600),  # (requests, seconds)
          "report": (12, 600), "wave": (30, 600), "sitrep": (6, 600), "driver": (120, 600), "admin": (60, 600)}
MCP_COSTLY_TOOLS = {"dispatch_relief", "parse_field_report", "train_agent"}
_hits: dict[tuple[str, str], list[float]] = {}
_hits_lock = threading.Lock()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def check_rate(request: Request, bucket: str):
    """Raises 429 with Retry-After when this client exceeded the bucket's limit."""
    limit, window = LIMITS[bucket]
    now, key = time.time(), (bucket, client_ip(request))
    with _hits_lock:
        recent = [t for t in _hits.get(key, []) if now - t < window]
        if len(recent) >= limit:
            retry = int(window - (now - recent[0])) + 1
            raise HTTPException(429, headers={"Retry-After": str(retry)},
                                detail=f"Rate limit: {limit} {bucket} requests per {window // 60} minutes "
                                       f"from one address. Try again in {retry} s.")
        recent.append(now)
        _hits[key] = recent


# ---- who is asking ---------------------------------------------------------------------------
def coordinator(request: Request) -> str:
    """Dependency for coordinator-only actions. With sign-in off (local, tests) the console is open."""
    try:
        return auth.coordinator(request.headers.get("authorization", "")) or "console"
    except auth.Denied as e:
        raise HTTPException(401, detail=str(e))


def is_coordinator(request: Request) -> bool:
    try:
        auth.coordinator(request.headers.get("authorization", ""))
        return True
    except auth.Denied:
        return False


@app.exception_handler(ops.OpsError)
def ops_conflict(request: Request, exc: ops.OpsError):
    return JSONResponse({"detail": str(exc)}, status_code=409)


class TrainReq(BaseModel):
    episodes: int = 5
    chaos: bool = True
    memory: dict | None = None  # browser's copy, restores learning after a serverless cold start


async def _media(f: UploadFile | None):
    if f is None or not f.filename:
        return None
    data = await f.read()
    return (data, f.content_type or "application/octet-stream") if data else None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True, "llm": agent.llm_available(), "models": agent._models(),
            "rl_policy": agent.policy.available(), "gemini_budget": agent.budget_status(),
            "storage": ops.store.backend(), "road_router": roads.enabled(), "coordinator_sign_in": auth.enabled()}


@app.get("/api/config")
def config():
    return {"firebase": auth.web_config()}


@app.get("/api/me")
def me(request: Request):
    try:
        email = auth.coordinator(request.headers.get("authorization", ""))
        return {"coordinator": True, "email": email, "sign_in": auth.enabled()}
    except auth.Denied as e:
        return {"coordinator": False, "email": None, "sign_in": True, "reason": str(e)}


@app.post("/api/dispatch")
async def dispatch(request: Request, text: str = Form(""), chaos: bool = Form(True), memory: str = Form(""),
                   image: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
    check_rate(request, "dispatch")
    try:
        agent.sync_memory(json.loads(memory) if memory else None)
    except ValueError:
        pass
    heals: list[str] = []
    img, aud = await _media(image), await _media(audio)
    report = await run_in_threadpool(agent.intake, text, img, aud, heals)
    result = await run_in_threadpool(
        lambda: agent.run_episode(extra_needs=report["needs"], extra_hazards=report["hazards"], chaos=chaos,
                                  strategy="llm" if agent.llm_available() else None, use_maps=True,
                                  explore=False))
    result["heals"] = heals + result["heals"]
    return {"intake": report, **result, "memory": agent.load_memory()}


@app.post("/api/train")
def train(req: TrainReq, request: Request):
    check_rate(request, "train")
    agent.sync_memory(req.memory)
    episodes, last = agent.train(max(1, min(20, req.episodes)), req.chaos)
    return {"episodes": episodes, "last": last, "memory": agent.load_memory()}


@app.get("/api/scenario")
def scenario():
    return agent.preview()


@app.get("/api/memory")
def memory():
    return agent.load_memory()


@app.post("/api/memory/reset")
def memory_reset(request: Request, who: str = Depends(coordinator)):
    check_rate(request, "reset")
    agent.reset_memory()
    return agent.load_memory()


# ---- live operations ---------------------------------------------------------------------
def _ops_out(st, request):
    return {"ops": ops.view(st, is_coordinator(request)), "summary": ops.summary(st)}


@app.get("/report")
def reporter_page():
    return FileResponse(STATIC / "report.html")


@app.get("/driver/{token}")
def driver_page(token: str):
    return FileResponse(STATIC / "driver.html")


@app.get("/track/{token}")
def track_page(token: str):
    return FileResponse(STATIC / "track.html")


@app.post("/api/reports")
async def submit_report(request: Request, text: str = Form(""), source: str = Form("console"),
                        lat: str = Form(""), lng: str = Form(""),
                        image: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
    """A field report from the console or the phone page: intake, then queue (or merge) as incidents."""
    check_rate(request, "report")
    img, aud = await _media(image), await _media(audio)
    if not (text.strip() or img or aud):
        raise HTTPException(400, detail="Send some text, a photo or a voice note.")
    heals: list[str] = []
    report = await run_in_threadpool(agent.intake, text, img, aud, heals)
    try:
        coords = (float(lat), float(lng)) if lat and lng else None
    except ValueError:
        coords = None
    out = await run_in_threadpool(ops.add_report, report, "phone" if source == "phone" else "console", coords)
    return {"incidents": out["incidents"], "merged": out["merged"], "track": out["track"],
            "track_url": f"/track/{out['track']}", "intake": report, "heals": heals,
            **_ops_out(out["ops"], request)}


@app.get("/api/ops")
def ops_state(request: Request):
    return _ops_out(ops.load(), request)


@app.post("/api/ops/wave")
def ops_wave(request: Request, who: str = Depends(coordinator)):
    check_rate(request, "wave")
    out = ops.run_wave(actor=who)
    return {**out, **_ops_out(out["ops"], request)}


@app.post("/api/ops/reset")
def ops_reset(request: Request, who: str = Depends(coordinator)):
    check_rate(request, "reset")
    return _ops_out(ops.reset(actor=who), request)


@app.post("/api/ops/sitrep")
def ops_sitrep(request: Request, who: str = Depends(coordinator)):
    check_rate(request, "sitrep")
    return ops.sitrep()


class DepotReq(BaseModel):
    name: str
    lat: float
    lng: float


class TruckReq(BaseModel):
    depot_id: str
    capacity: int = 250


class TruckPatch(BaseModel):
    capacity: int | None = None
    in_service: bool | None = None


class RestockReq(BaseModel):
    depot_id: str
    item: str
    qty: int
    note: str = ""


class ReasonReq(BaseModel):
    reason: str = ""


def _admin(request, who, fn, *args):
    check_rate(request, "admin")
    fn(*args, actor=who)
    return _ops_out(ops.load(), request)


@app.post("/api/ops/depots")
def add_depot(req: DepotReq, request: Request, who: str = Depends(coordinator)):
    return _admin(request, who, ops.add_depot, req.name, req.lat, req.lng)


@app.post("/api/ops/trucks")
def add_truck(req: TruckReq, request: Request, who: str = Depends(coordinator)):
    return _admin(request, who, ops.add_truck, req.depot_id, req.capacity)


@app.patch("/api/ops/trucks/{truck_id}")
def edit_truck(truck_id: str, req: TruckPatch, request: Request, who: str = Depends(coordinator)):
    return _admin(request, who, ops.set_truck, truck_id, req.capacity, req.in_service)


@app.post("/api/ops/restock")
def restock(req: RestockReq, request: Request, who: str = Depends(coordinator)):
    return _admin(request, who, ops.restock, req.depot_id, req.item, req.qty, req.note)


@app.post("/api/ops/hazards/{hazard_id}/clear")
def clear_hazard(hazard_id: str, request: Request, who: str = Depends(coordinator)):
    return _admin(request, who, ops.clear_hazard, hazard_id)


@app.post("/api/ops/incidents/{incident_id}/close")
def close_incident(incident_id: str, req: ReasonReq, request: Request, who: str = Depends(coordinator)):
    return _admin(request, who, ops.close_incident, incident_id, req.reason)


@app.post("/api/ops/routes/{route_id}/recall")
def recall_route(route_id: str, request: Request, who: str = Depends(coordinator)):
    check_rate(request, "admin")
    out = ops.recall_route(route_id, actor=who)
    return {"learned": out["learned"], **_ops_out(ops.load(), request)}


# ---- drivers and reporters (no sign-in: the link itself is the secret) ----------------------------
class StopReq(BaseModel):
    outcome: str
    qty: int | None = None
    note: str = ""
    lat: float | None = None
    lng: float | None = None


@app.get("/api/driver/{token}")
def driver_state(token: str):
    out = ops.driver_view(token)
    if out is None:
        raise HTTPException(404, detail="Unknown driver link. Ask the coordinator for your truck's link.")
    return out


@app.post("/api/driver/{token}/stops/{n}")
def driver_stop(token: str, n: int, req: StopReq, request: Request):
    check_rate(request, "driver")
    if ops.driver_view(token) is None:
        raise HTTPException(404, detail="Unknown driver link.")
    coords = (req.lat, req.lng) if req.lat is not None and req.lng is not None else None
    out = ops.driver_update(token, n, req.outcome, req.qty, req.note, coords)
    return {"learned": out["learned"], "closed": bool(out["closed"]), **ops.driver_view(token)}


@app.get("/api/track/{token}")
def track_state(token: str):
    out = ops.track_view(token)
    if out is None:
        raise HTTPException(404, detail="Unknown tracking link.")
    return out


@app.post("/mcp")
async def mcp(request: Request):
    """MCP Streamable HTTP endpoint (stateless, JSON responses)."""
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(mcp_server.error(None, -32700, "Parse error"), status_code=400)
    messages = payload if isinstance(payload, list) else [payload]
    costly = [m for m in messages if isinstance(m, dict) and m.get("method") == "tools/call"
              and isinstance(m.get("params"), dict) and m["params"].get("name") in MCP_COSTLY_TOOLS]
    for m in costly:
        try:
            check_rate(request, "mcp")
        except HTTPException as e:
            return JSONResponse(mcp_server.error(m.get("id"), -32000, e.detail), status_code=429, headers=e.headers)
    if isinstance(payload, list):
        out = [r for r in [await run_in_threadpool(mcp_server.handle, m) for m in payload] if r is not None]
        return JSONResponse(out) if out else Response(status_code=202)
    out = await run_in_threadpool(mcp_server.handle, payload)
    return JSONResponse(out) if out is not None else Response(status_code=202)


@app.get("/mcp")
def mcp_stream():
    return Response(status_code=405)  # stateless server: no server-initiated SSE stream


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
