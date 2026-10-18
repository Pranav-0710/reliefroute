"""ReliefRoute API: FastAPI server for the multimodal disaster-relief logistics agent."""
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import agent
import mcp_server

app = FastAPI(title="ReliefRoute")
STATIC = Path(__file__).parent / "static"

# ---- rate limits ----------------------------------------------------------------------
# Per client IP, sliding window, per server instance. Protects the shared Gemini quota and memory
# of a public deployment; not a substitute for authentication.
LIMITS = {"dispatch": (8, 600), "train": (60, 600), "mcp": (10, 600), "reset": (5, 600)}  # (requests, seconds)
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
            "rl_policy": agent.policy.available(), "gemini_budget": agent.budget_status()}


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
def memory_reset(request: Request):
    check_rate(request, "reset")
    agent.reset_memory()
    return agent.load_memory()


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
