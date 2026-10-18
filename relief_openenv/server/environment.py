"""OpenEnv Environment wrapping the ReliefRoute flood simulator (env.py)."""
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for env.py

from env import ReliefEnv  # noqa: E402

from ..models import ReliefAction, ReliefObservation, ReliefState  # noqa: E402


class ReliefEnvironment(Environment):
    """Flood-relief dispatch: assign requests to depots and trucks over up to 3 rounds.

    Reward per round is the simulator reward (deliveries weighted by urgency, minus delays,
    blocked legs and unmet needs); the final observation also carries score (% of best possible).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True  # every session gets its own ReliefEnv instance

    def __init__(self):
        super().__init__()
        self._env = ReliefEnv()
        self._state = ReliefState(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, chaos: bool = True,
              n_needs: int = 5, **kwargs: Any) -> ReliefObservation:
        obs = self._env.reset(seed=seed, chaos=chaos, n_needs=max(1, min(20, int(n_needs))))
        st = self._env.state()
        self._state = ReliefState(episode_id=episode_id or str(uuid.uuid4()), step_count=0, seed=seed,
                                  total_reward=0.0, max_reward=st["max_reward"], deliveries=0)
        return ReliefObservation(done=False, reward=0.0, **obs)

    def step(self, action: ReliefAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ReliefObservation:
        res = self._env.step({"assignments": [a.model_dump(exclude_none=True) for a in action.assignments]})
        st = self._env.state()
        self._state.step_count = st["step"]
        self._state.total_reward = st["total_reward"]
        self._state.deliveries = len(st["deliveries"])
        score = round(max(0.0, 100 * st["total_reward"] / st["max_reward"]), 1) if res["done"] and st["max_reward"] else None
        return ReliefObservation(done=res["done"], reward=res["reward"], info=res["info"], score=score,
                                 **res["observation"])

    @property
    def state(self) -> ReliefState:
        return self._state
