# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hackathon Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LogiChainAction, LogiChainObservation


class HackathonEnv(
    EnvClient[LogiChainAction, LogiChainObservation, State]
):
    """
    Client for the Hackathon Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with HackathonEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.dashboard_text)
        ...
        ...     result = client.step(LogiChainAction(type="call_tool", tool_name="query_network"))
        ...     print(result.observation.tool_result)
    """

    def _step_payload(self, action: LogiChainAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[LogiChainObservation]:
        obs_data = payload.get("observation", {})
        observation = LogiChainObservation(
            dashboard_text=obs_data.get("dashboard_text", ""),
            alerts=obs_data.get("alerts", []),
            available_tools=obs_data.get("available_tools", []),
            episode_id=obs_data.get("episode_id", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            time_step=obs_data.get("time_step", 0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
