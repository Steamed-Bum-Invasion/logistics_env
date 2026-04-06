from typing import List, Optional, Dict, Any
from openenv.core.env_server import Action, Observation, State


class LogiChainAction(Action):
    """
    Batched action (action economy).
    Each step contains a list of atomic tool-like operations.
    """

    actions: List[Dict[str, Any]]  # each dict = atomic action


class LogiChainObservation(Observation):
    """
    Messy ERP-style observation.
    """

    dashboard_text: str
    alerts: List[str]
    available_actions: List[str]
    remaining_actions: int


class LogiChainState(State):
    """
    Ground truth (not fully exposed to agent).
    """

    time_step: int = 0
    max_steps: int = 50

    graph: Dict[str, Dict[str, int]] = {}

    drivers: Dict[str, Dict[str, Any]] = {}
    orders: Dict[str, Dict[str, Any]] = {}

    traffic: Dict[str, float] = {}

    metrics: Dict[str, int] = {
        "completed": 0,
        "failed": 0,
        "on_time": 0,
    }
