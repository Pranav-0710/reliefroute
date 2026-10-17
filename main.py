"""ReliefRoute API: FastAPI server for the multimodal disaster-relief logistics agent."""
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import agent

app = FastAPI(title="ReliefRoute")
STATIC = Path(__file__).parent / "static"


class TrainReq(BaseModel):
    episodes: int = 5
    chaos: bool = True


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
async def dispatch(text: str = Form(""), chaos: bool = Form(True),
                   image: UploadFile | None = File(None), audio: UploadFile | None = File(None)):
    heals: list[str] = []
    img, aud = await _media(image), await _media(audio)
    report = agent.intake(text, img, aud, heals)
    result = agent.run_episode(extra_needs=report["needs"], extra_hazards=report["hazards"], chaos=chaos,
                               strategy="llm" if agent.llm_available() else None, use_maps=True,
                               explore=False)
    result["heals"] = heals + result["heals"]
    return {"intake": report, **result}


@app.post("/api/train")
def train(req: TrainReq):
    return {"episodes": agent.train(max(1, min(20, req.episodes)), req.chaos), "memory": agent.load_memory()}


@app.get("/api/memory")
def memory():
    return agent.load_memory()


@app.post("/api/memory/reset")
def memory_reset():
    agent.reset_memory()
    return agent.load_memory()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
