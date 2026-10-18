"""Typed OpenEnv client for the ReliefRoute environment.

    from relief_openenv import Assignment, ReliefAction, ReliefEnvClient

    with ReliefEnvClient(base_url="http://localhost:8001").sync() as env:
        result = env.reset(seed=7)
        obs = result.observation
        plan = [Assignment(need_id=n["id"], depot_id="D1") for n in obs.pending_needs]
        result = env.step(ReliefAction(assignments=plan))
"""
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import ReliefAction, ReliefObservation, ReliefState


class ReliefEnvClient(EnvClient[ReliefAction, ReliefObservation, ReliefState]):
    def _step_payload(self, action: ReliefAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ReliefObservation]:
        fields = {k: v for k, v in payload.get("observation", {}).items() if k not in ("done", "reward")}
        obs = ReliefObservation(done=payload.get("done", False), reward=payload.get("reward"), **fields)
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict[str, Any]) -> ReliefState:
        return ReliefState(**payload)
