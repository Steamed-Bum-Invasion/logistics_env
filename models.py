"""Client-side state and observation types for the LogiChain Env environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server import State
from openenv.core.env_server.types import Observation, Action
from pydantic import Field

AVAILABLE_TOOLS = [
    "assign_order",
    "reroute_driver",
    "delay_order",
    "escalate_order",
    "query_driver",
    "query_order",
    "query_network",
]


class LogiChainAction(Action):
    """Action for the LogiChain environment.

    Wraps MCP-style tool calls with a discriminator type.
    """

    type: Literal["call_tool", "list_tools"] = Field(
        default="call_tool", description="Action type discriminator"
    )
    tool_name: str = Field(default="", description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


class LogiChainState(State):
    """Runtime state exposed by the environment server."""

    episode_id: str = Field(default="", description="Unique episode identifier")
    time_step: int = Field(default=0, description="Current simulation time step")
    step_count: int = Field(default=0, description="Total agent actions taken")
    orders_pending: int = Field(default=0, description="Orders waiting to be assigned")
    orders_in_transit: int = Field(
        default=0, description="Orders currently being delivered"
    )
    orders_delivered: int = Field(
        default=0, description="Successfully delivered orders"
    )
    orders_failed: int = Field(default=0, description="Failed/expired orders")
    on_time_deliveries: int = Field(
        default=0, description="Orders delivered before deadline"
    )


class LogiChainObservation(Observation):
    """Initial observation returned by reset(). Contains the shift dashboard text."""

    dashboard_text: str = Field(default="", description="Dashboard state")
    alerts: List[str] = Field(default_factory=list, description="Active alerts")
    available_tools: List[str] = Field(
        default_factory=list, description="Available tools"
    )
    episode_id: str = Field(default="", description="Episode ID")
    time_step: int = Field(default=0, description="Current simulation time step")


class LogiChainToolObservation(Observation):
    """Observation returned for every tool-call step."""

    tool_name: str = Field(default="", description="Name of the tool that was called")
    tool_result: str = Field(default="", description="Text result from the tool")
    error_msg: Optional[str] = Field(
        default=None, description="Error message if call failed"
    )
    alerts: List[str] = Field(default_factory=list, description="Active alerts")
    available_tools: List[str] = Field(
        default_factory=list, description="Available tools"
    )
    time_step: int = Field(default=0, description="Current simulation time step")
