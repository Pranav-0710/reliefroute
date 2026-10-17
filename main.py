"""ReliefRoute API: FastAPI server for the multimodal disaster-relief logistics agent."""
import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import agent
import mcp_server

app = FastAPI(title="ReliefRoute")
STATIC = Path(__file__).parent / "static"


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
    return {"ok": True, "llm": agent.llm_available(), "models": agent._models()}


@app.post("/api/dispatch")
async def dispatch(text: str = Form(""), chaos: bool = Form(True), memory: str = Form(""),
                   image: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
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
def train(req: TrainReq):
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
def memory_reset():
    agent.reset_memory()
    return agent.load_memory()


@app.post("/mcp")
async def mcp(request: Request):
    """MCP Streamable HTTP endpoint (stateless, JSON responses)."""
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(mcp_server.error(None, -32700, "Parse error"), status_code=400)
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
