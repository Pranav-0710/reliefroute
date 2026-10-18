"""Typed action, observation and state for the ReliefRoute OpenEnv environment."""
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class Assignment(BaseModel):
    need_id: str
    depot_id: str
    detour: bool = False
    truck_id: Optional[str] = None


class ReliefAction(Action):
    """One dispatch round. List order is loading order; each truck visits its stops in that order."""
    assignments: List[Assignment] = Field(default_factory=list)


class ReliefObservation(Observation):
    step: int = 0
    max_steps: int = 3
    depots: List[Dict[str, Any]] = Field(default_factory=list)
    truck_capacity: int = 0
    pending_needs: List[Dict[str, Any]] = Field(default_factory=list)
    known_hazards: List[Dict[str, Any]] = Field(default_factory=list)
    route_matrix: Dict[str, Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    info: Dict[str, Any] = Field(default_factory=dict, description="Outcome of the last round: deliveries, blocks, trucks")
    score: Optional[float] = Field(default=None, description="% of best possible outcome, set when done")


class ReliefState(State):
    seed: Optional[int] = None
    total_reward: float = 0.0
    max_reward: float = 0.0
    deliveries: int = 0
