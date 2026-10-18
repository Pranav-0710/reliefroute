"""OpenEnv HTTP/WebSocket server for the ReliefRoute environment.

Run from the repository root:
    uvicorn relief_openenv.server.app:app --port 8001
"""
from openenv.core.env_server.http_server import create_app

from ..models import ReliefAction, ReliefObservation, ReliefState
from .environment import ReliefEnvironment

app = create_app(ReliefEnvironment, ReliefAction, ReliefObservation, env_name="reliefroute",
                 max_concurrent_envs=16, state_cls=ReliefState)
