# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
LogiChain Environment — OpenEnv Environment implementation.

Tool dispatch is delegated to an internal _LogiChainMCPDelegate (MCPEnvironment)
to retain FastMCP schema generation and MCP-level argument validation.
LogiChainEnvironment controls reward computation and returns LogiChainToolObservation
— a plain Pydantic model — so the reward field survives the WebSocket
serialization round-trip intact.
"""

import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastmcp import FastMCP
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
)
from openenv.core.env_server.types import Observation, Action

from logistics_env.models import (
    AVAILABLE_TOOLS,
    LogiChainAction,
    LogiChainState,
    LogiChainObservation,
    LogiChainToolObservation,
)
from logistics_env.server.network_graph import NetworkGraph
from logistics_env.server import tools
from logistics_env.server import rewards


def _load_yaml(filename: str, server_dir: Path) -> dict:
    """Load YAML config from server directory."""
    config_path = server_dir / filename
    with open(config_path) as f:
        return yaml.safe_load(f)


class _LogiChainMCPDelegate(MCPEnvironment):
    """
    Thin MCPEnvironment wrapper that holds the FastMCP server.

    Provides _handle_call_tool() and _handle_list_tools() to
    LogiChainEnvironment without any reward logic.
    """

    def reset(self, **kwargs: Any) -> Observation:
        return Observation(done=False, reward=0.0)

    def _step_impl(self, action: Action, **kwargs: Any) -> Observation:
        return Observation(done=False, reward=0.0)

    @property
    def state(self) -> LogiChainState:
        return LogiChainState()


class LogiChainEnvironment(Environment):
    """
    Logistics chain management environment.

    The agent manages a fleet of drivers delivering orders across a network.
    New orders arrive dynamically. The agent must assign drivers, route them,
    and handle delays/escalations.

    Time advances automatically every step(). Orders fail when deadline exceeded.
    Drivers move incrementally toward their destinations.

    Tools:
        assign_order: Assign a pending order to a driver
        reroute_driver: Reroute a driver to their order's dropoff
        delay_order: Extend an order's deadline by 3 steps
        escalate_order: Escalate an order (extend deadline by 5, costs more)
        query_driver: Get driver status and location
        query_order: Get order status and details
        query_network: Get network topology and traffic info
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_name: str = None):
        self._server_dir = Path(os.environ.get("SERVER_DIR", "/app/env/server"))
        self._network: Optional[NetworkGraph] = None
        self._drivers: Dict[str, Dict[str, Any]] = {}
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[str] = []
        self._step_count = 0
        self._rng = random.Random()
        self._state = LogiChainState()
        self._reset_count = 0

        self._reward_config = _load_yaml("reward_config.yaml", self._server_dir)
        self._task_config = _load_yaml("task_config.yaml", self._server_dir)
        self._current_task = task_name or self._task_config.get("default_task", "quick_delivery")

        self._steps_without_progress = 0
        self._last_progress_step = 0
        self._total_deliveries_at_start = 0
        self._total_assignments_at_start = 0

        self._num_drivers = 5
        self._initial_orders = 10
        self._order_spawn_rate = 0.2
        self._order_deadline_min = 10
        self._order_deadline_max = 30
        self._traffic_spike_rate = 0.15
        self._traffic_spike_multiplier = 1.3

        mcp = FastMCP("logichain_env")

        @mcp.tool
        def assign_order(order_id: str, driver_id: str) -> str:
            """Assign a pending order to a driver."""
            return tools.assign_order(self, order_id, driver_id)

        @mcp.tool
        def reroute_driver(driver_id: str) -> str:
            """Reroute a driver to their assigned order's dropoff location."""
            return tools.reroute_driver(self, driver_id)

        @mcp.tool
        def delay_order(order_id: str) -> str:
            """Extend an order's deadline by 3 steps."""
            return tools.delay_order(self, order_id)

        @mcp.tool
        def escalate_order(order_id: str) -> str:
            """Escalate an order with priority handling (extend deadline by 5 steps)."""
            return tools.escalate_order(self, order_id)

        @mcp.tool
        def query_driver(driver_id: str) -> str:
            """Get detailed status of a driver."""
            return tools.query_driver(self, driver_id)

        @mcp.tool
        def query_order(order_id: str) -> str:
            """Get detailed status of an order."""
            return tools.query_order(self, order_id)

        @mcp.tool
        def query_network() -> str:
            """Get network topology and current traffic conditions."""
            return tools.query_network(self)

        super().__init__()
        self._mcp_env = _LogiChainMCPDelegate(mcp)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if seed is not None:
            self._rng.seed(seed)

        self._reset_count += 1
        self._alerts = []
        self._step_count = 0
        self._state.time_step = 0
        self._state.episode_id = episode_id or str(uuid.uuid4())

        self._steps_without_progress = 0
        self._last_progress_step = 0
        self._total_deliveries_at_start = 0
        self._total_assignments_at_start = 0

        task_cfg = self._task_config.get("tasks", {}).get(self._current_task, {})
        self._load_simulation_config(task_cfg)
        self._network = self._create_network(seed, task_cfg)
        self._drivers = self._spawn_drivers(self._num_drivers)
        self._orders = self._generate_orders(self._initial_orders)

        return LogiChainObservation(
            dashboard_text=self._render_dashboard(),
            alerts=list(self._alerts),
            available_tools=list(AVAILABLE_TOOLS),
            episode_id=self._state.episode_id,
            done=False,
            reward=0.0,
            time_step=self._state.time_step,
        )

    def _load_simulation_config(self, task_cfg: Dict[str, Any]) -> None:
        """Load simulation parameters from task config."""
        sim_cfg = task_cfg.get("simulation", {})
        self._num_drivers = sim_cfg.get("num_drivers", 5)
        self._initial_orders = sim_cfg.get("initial_orders", 10)
        self._order_spawn_rate = sim_cfg.get("order_spawn_rate", 0.2)
        self._order_deadline_min = sim_cfg.get("order_deadline_min", 10)
        self._order_deadline_max = sim_cfg.get("order_deadline_max", 30)
        self._traffic_spike_rate = sim_cfg.get("traffic_spike_rate", 0.15)
        self._traffic_spike_multiplier = sim_cfg.get("traffic_spike_multiplier", 1.3)

    def _create_network(self, seed: Optional[int], task_cfg: Dict[str, Any]) -> NetworkGraph:
        """Create network from task configuration."""
        network_cfg = task_cfg.get("network", {})
        variation_mode = network_cfg.get("variation_mode", "fixed")

        if variation_mode == "per_reset":
            network_seed = (seed or 0) + self._reset_count * 1000
        else:
            network_seed = seed

        if "config_file" in network_cfg:
            config_path = self._server_dir / network_cfg["config_file"]
            return NetworkGraph.from_config(str(config_path), seed=network_seed)
        elif network_cfg:
            return NetworkGraph.from_dict(network_cfg, seed=network_seed)
        else:
            return NetworkGraph(seed=network_seed)

    def step(
        self,
        action: LogiChainAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count += 1
        self._state.time_step += 1

        self._move_drivers()
        self._update_orders()
        self._spawn_orders()
        self._update_traffic()

        if action.type == "list_tools":
            return self._mcp_env._handle_list_tools()

        if action.type == "call_tool":
            call_action = CallToolAction(
                tool_name=action.tool_name, arguments=action.arguments
            )
            return self._handle_call_tool(call_action, timeout_s=timeout_s)

        return LogiChainToolObservation(
            done=False,
            reward=0.0,
            tool_name="",
            tool_result="",
            error_msg=f"Unknown action type: {action.type}. Use 'call_tool' or 'list_tools'.",
            alerts=list(self._alerts),
            available_tools=list(AVAILABLE_TOOLS),
            time_step=self._state.time_step,
        )

    def _handle_call_tool(
        self,
        action: CallToolAction,
        timeout_s: Optional[float] = None,
    ) -> LogiChainToolObservation:
        """
        Dispatch a CallToolAction via the internal MCPEnvironment, compute
        reward, and return a LogiChainToolObservation with a plain float reward.
        """
        call_obs: CallToolObservation = self._mcp_env._handle_call_tool(
            action, timeout_s=timeout_s
        )
        result_text = self._extract_result_text(call_obs)

        is_error = call_obs.error is not None or result_text.startswith("ERROR")

        logi_action = LogiChainAction(
            type="call_tool",
            tool_name=action.tool_name,
            arguments=action.arguments,
        )

        reward = rewards.compute_step_reward(self, logi_action, result_text, is_error)
        done = self._is_done()

        error_msg: Optional[str] = None
        if call_obs.error is not None:
            err = call_obs.error
            error_msg = err.message if hasattr(err, "message") else str(err)

        return LogiChainToolObservation(
            tool_name=action.tool_name,
            tool_result=result_text,
            error_msg=error_msg,
            done=done,
            reward=reward,
            alerts=list(self._alerts),
            available_tools=list(AVAILABLE_TOOLS),
            time_step=self._state.time_step,
        )

    @staticmethod
    def _extract_result_text(call_obs: CallToolObservation) -> str:
        """Extract plain text from a CallToolObservation."""
        if call_obs.error is not None:
            err = call_obs.error
            msg = err.message if hasattr(err, "message") else str(err)
            return f"ERROR: {msg}"

        r = call_obs.result
        if r is None:
            return ""
        if hasattr(r, "data") and r.data is not None:
            return str(r.data)
        if hasattr(r, "content") and r.content:
            first = r.content[0]
            return first.text if hasattr(first, "text") else str(first)
        return str(r)

    @property
    def state(self) -> LogiChainState:
        pending = sum(1 for o in self._orders.values() if o["status"] == "pending")
        in_transit = sum(
            1
            for o in self._orders.values()
            if o["status"] in ("assigned", "in_transit")
        )
        delivered = sum(1 for o in self._orders.values() if o["status"] == "delivered")
        failed = sum(1 for o in self._orders.values() if o["status"] == "failed")
        on_time = sum(
            1
            for o in self._orders.values()
            if o["status"] == "delivered" and o.get("delivered", 999) <= o["deadline"]
        )

        return LogiChainState(
            episode_id=getattr(self._state, "episode_id", ""),
            step_count=self._step_count,
            time_step=self._state.time_step,
            orders_pending=pending,
            orders_in_transit=in_transit,
            orders_delivered=delivered,
            orders_failed=failed,
            on_time_deliveries=on_time,
        )

    def _move_drivers(self) -> int:
        """Move drivers 1 unit along their route. Returns number of deliveries.
        
        ETA logic: Driver arrives when ETA goes negative (eta < 0), then the
        next location in the route is popped. The ETA is then recalculated
        for the next segment of the route.
        """
        deliveries = 0
        for d in self._drivers.values():
            if d["status"] == "moving":
                d["eta"] -= 1
                if d["eta"] < 0 and d["route"]:
                    d["location"] = d["route"].pop(0)
                    if not d["route"]:
                        d["status"] = "idle"
                        deliveries += 1
                    else:
                        next_stop = d["route"][0]
                        cost, _ = self._network.shortest_path(d["location"], next_stop)
                        d["eta"] = cost
        return deliveries

    def _update_orders(self):
        """Update order statuses: check deliveries and deadline failures."""
        for oid, o in self._orders.items():
            if o["status"] == "assigned":
                driver = self._drivers[o["assigned"]]
                if driver["status"] == "moving":
                    o["status"] = "in_transit"
                elif driver["location"] == o["dropoff"] and driver["status"] == "idle":
                    o["status"] = "delivered"
                    o["delivered"] = self._state.time_step
            elif o["status"] == "in_transit":
                driver = self._drivers[o["assigned"]]
                if driver["location"] == o["dropoff"] and driver["status"] == "idle":
                    o["status"] = "delivered"
                    o["delivered"] = self._state.time_step
            elif o["status"] == "pending" and self._state.time_step > o["deadline"]:
                o["status"] = "failed"
                o["failed_at"] = self._state.time_step

    def _spawn_orders(self):
        """Spawn new orders with configurable probability per step."""
        if self._rng.random() < self._order_spawn_rate:
            new_orders = self._generate_orders(self._rng.randint(1, 2))
            self._orders.update(new_orders)
            for oid in new_orders:
                o = self._orders[oid]
                self._alerts.append(
                    f"New order: {oid}: {o['pickup']} -> {o['dropoff']} (deadline: {o['deadline']})"
                )

    def _update_traffic(self):
        """Randomly trigger traffic spikes with configurable rate."""
        if self._rng.random() < self._traffic_spike_rate:
            self._alerts.append("Traffic spike detected")
            self._network.update_traffic(multiplier=self._traffic_spike_multiplier)

    def _is_done(self) -> bool:
        """Check if episode should end based on task conditions."""
        if all(o["status"] in ("delivered", "failed") for o in self._orders.values()):
            return True

        if self._state.time_step >= 500:
            return True

        cfg = self._reward_config.get("no_progress", {})
        max_no_progress = cfg.get("max_no_progress", 50)
        if self._steps_without_progress > max_no_progress:
            return True

        task = self._task_config.get("tasks", {}).get(self._current_task, {})
        done_condition = task.get("done_condition", {})

        if done_condition.get("type") == "time_limit":
            max_time = done_condition.get("max_time_steps", 30)
            if self._state.time_step >= max_time:
                return True

        return False

    def _spawn_drivers(self, n: int) -> Dict[str, Dict[str, Any]]:
        """Spawn drivers at random hub nodes."""
        hubs = self._network.get_hub_nodes()
        if not hubs:
            hubs = [self._network.get_all_nodes()[0]]

        drivers = {}
        for i in range(n):
            start_location = self._rng.choice(hubs)
            drivers[f"D{i}"] = {
                "location": start_location,
                "route": [],
                "eta": 0,
                "status": "idle",
                "orders": [],
            }
        return drivers

    def _generate_orders(self, n: int) -> Dict[str, Dict[str, Any]]:
        """Generate orders with configurable deadline range."""
        nodes = self._network.get_all_nodes()
        orders = {}
        existing_count = len(self._orders)

        for i in range(n):
            pickup = self._rng.choice(nodes)
            dropoff = self._rng.choice(nodes)
            while dropoff == pickup:
                dropoff = self._rng.choice(nodes)

            deadline = self._state.time_step + self._rng.randint(
                self._order_deadline_min, self._order_deadline_max
            )

            oid = f"O{existing_count + i}"
            orders[oid] = {
                "pickup": pickup,
                "dropoff": dropoff,
                "deadline": deadline,
                "status": "pending",
                "assigned": None,
                "created": self._state.time_step,
                "delivered": None,
            }
        return orders

    def _render_dashboard(self) -> str:
        lines = ["=== DRIVERS ==="]
        for k, d in self._drivers.items():
            lines.append(f"{k}: {d['status']} @ {d['location']}")

        lines.append("\n=== ORDERS ===")
        for k, o in self._orders.items():
            lines.append(f"{k}: {o['status']} deadline={o['deadline']}")

        lines.append(f"\nTime: {self._state.time_step}")

        return "\n".join(lines)
