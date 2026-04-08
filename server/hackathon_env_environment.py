# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: need to understand how to use pyright.
# pyright: reportUnusedImport=false
#
# ERROR: I CANT SEEM TO DO LSPRESTRT. NEED TO CHECK
"""
LogiChain Environment — OpenEnv Environment implementation.

Tool dispatch is delegated to an internal _LogiChainMCPDelegate (MCPEnvironment)
to retain FastMCP schema generation and MCP-level argument validation.
LogiChainEnvironment controls reward computation and returns LogiChainToolObservation
— a plain Pydantic model — so the reward field survives the WebSocket
serialization round-trip intact.
"""

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

try:
    from ..models import (
        AVAILABLE_TOOLS,
        LogiChainAction,
        LogiChainState,
        LogiChainObservation,
        LogiChainToolObservation,
    )
    from .network_graph import NetworkGraph
    from .grader import TaskGrader
except ImportError:
    from models import (
        AVAILABLE_TOOLS,
        LogiChainAction,
        LogiChainState,
        LogiChainObservation,
        LogiChainToolObservation,
    )
    from server.network_graph import NetworkGraph
    from server.grader import TaskGrader


def _load_yaml(filename: str) -> dict:
    """Load YAML config from server directory."""
    server_dir = Path(__file__).parent
    config_path = server_dir / filename
    with open(config_path) as f:
        return yaml.safe_load(f)


class _LogiChainMCPDelegate(MCPEnvironment):
    """
    Thin MCPEnvironment wrapper that holds the FastMCP server.

    Provides _handle_call_tool() and _handle_list_tools() to
    LogiChainEnvironment without any reward logic.
    """

    def reset(self, **kwargs: Any) -> Observation:  # type: ignore[override]
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
        self._network: Optional[NetworkGraph] = None
        self._drivers: Dict[str, Dict[str, Any]] = {}
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[str] = []
        self._step_count = 0
        self._rng = __import__("random").Random()
        self._state = LogiChainState()

        # Load reward config
        self._reward_config = _load_yaml("reward_config.yaml")

        # Load task config and set task
        self._task_config = _load_yaml("task_config.yaml")
        self._current_task = task_name or self._task_config.get("default_task", "quick_delivery")

        # Initialize grader
        self._grader = TaskGrader(self._task_config)
        self._grader.set_task(self._current_task)

        # Progress tracking for no-progress penalty
        self._steps_without_progress = 0
        self._last_progress_step = 0
        self._total_deliveries_at_start = 0
        self._total_assignments_at_start = 0

        mcp = FastMCP("logichain_env")

        @mcp.tool
        def assign_order(order_id: str, driver_id: str) -> str:
            """
            Assign a pending order to a driver.

            Args:
                order_id: Order identifier (e.g. 'O0', 'O1'). Use query_order to see valid IDs.
                driver_id: Driver identifier (e.g. 'D0', 'D1'). Use query_driver to see valid IDs.

            Returns:
                Confirmation with route info, or error if order/driver invalid
            """
            return self._assign_order_impl(order_id, driver_id)

        @mcp.tool
        def reroute_driver(driver_id: str) -> str:
            """
            Reroute a driver to their assigned order's dropoff location.

            Args:
                driver_id: Driver identifier. Use query_driver to see valid IDs.

            Returns:
                Updated route info, or error if driver has no orders
            """
            return self._reroute_driver_impl(driver_id)

        @mcp.tool
        def delay_order(order_id: str) -> str:
            """
            Extend an order's deadline by 3 steps.

            Args:
                order_id: Order identifier. Use query_order to see valid IDs.

            Returns:
                Confirmation with new deadline, or error if order invalid
            """
            return self._delay_order_impl(order_id)

        @mcp.tool
        def escalate_order(order_id: str) -> str:
            """
            Escalate an order with priority handling (extend deadline by 5 steps).

            Args:
                order_id: Order identifier. Use query_order to see valid IDs.

            Returns:
                Confirmation with new deadline, or error if order invalid
            """
            return self._escalate_order_impl(order_id)

        @mcp.tool
        def query_driver(driver_id: str) -> str:
            """
            Get detailed status of a driver.

            Args:
                driver_id: Driver identifier (e.g. 'D0', 'D1', 'D2', 'D3', 'D4')

            Returns:
                Driver status including location, route, assigned orders, and ETA
            """
            return self._query_driver_impl(driver_id)

        @mcp.tool
        def query_order(order_id: str) -> str:
            """
            Get detailed status of an order.

            Args:
                order_id: Order identifier (e.g. 'O0', 'O1'). Check dashboard for valid IDs.

            Returns:
                Order details including pickup, dropoff, deadline, and current status
            """
            return self._query_order_impl(order_id)

        @mcp.tool
        def query_network() -> str:
            """
            Get network topology and current traffic conditions.

            Returns:
                Network map with nodes, connections, and traffic multipliers
            """
            return self._query_network_impl()

        super().__init__()
        self._mcp_env = _LogiChainMCPDelegate(mcp)

    # ── OpenEnv Interface ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        if seed is not None:
            self._rng.seed(seed)

        self._alerts = []
        self._step_count = 0
        self._state.time_step = 0
        self._state.episode_id = episode_id or str(uuid.uuid4())

        # Reset progress tracking
        self._steps_without_progress = 0
        self._last_progress_step = 0
        self._total_deliveries_at_start = 0
        self._total_assignments_at_start = 0

        self._network = NetworkGraph(seed=seed)
        self._drivers = self._spawn_drivers(5)
        self._orders = self._generate_orders(10)

        return LogiChainObservation(
            dashboard_text=self._render_dashboard(),
            alerts=list(self._alerts),
            available_tools=list(AVAILABLE_TOOLS),
            episode_id=self._state.episode_id,
            done=False,
            reward=0.0,
            time_step=self._state.time_step,
        )

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

        # Convert CallToolAction to LogiChainAction for reward computation
        logi_action = LogiChainAction(
            type="call_tool",
            tool_name=action.tool_name,
            arguments=action.arguments,
        )

        reward = self._compute_step_reward(logi_action, result_text, is_error)
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

    # ── Tool Implementations ───────────────────────────────────────────────

    def _assign_order_impl(self, order_id: str, driver_id: str) -> str:
        if order_id not in self._orders:
            return f"ERROR: Order '{order_id}' not found"
        if driver_id not in self._drivers:
            return f"ERROR: Driver '{driver_id}' not found"

        order = self._orders[order_id]
        driver = self._drivers[driver_id]

        if order["status"] != "pending":
            return (
                f"ERROR: Order '{order_id}' is not pending (status: {order['status']})"
            )

        cost, path = self._network.shortest_path(driver["location"], order["pickup"])

        driver["route"] = path[1:] if path and path[0] == driver["location"] else path
        driver["eta"] = cost
        driver["status"] = "moving"
        driver["orders"].append(order_id)

        order["status"] = "assigned"
        order["assigned"] = driver_id

        return f"OK: Assigned {order_id} to {driver_id}. Route: {' -> '.join(path)}. ETA: {cost}"

    def _reroute_driver_impl(self, driver_id: str) -> str:
        if driver_id not in self._drivers:
            return f"ERROR: Driver '{driver_id}' not found"

        driver = self._drivers[driver_id]
        if not driver["orders"]:
            return f"ERROR: Driver '{driver_id}' has no assigned orders"

        oid = driver["orders"][0]
        order = self._orders[oid]

        cost, path = self._network.shortest_path(driver["location"], order["dropoff"])
        driver["route"] = path[1:] if path and path[0] == driver["location"] else path
        driver["eta"] = cost
        driver["status"] = "moving"

        return f"OK: Rerouted {driver_id} to {order['dropoff']}. Route: {' -> '.join(path)}. ETA: {cost}"

    def _delay_order_impl(self, order_id: str) -> str:
        if order_id not in self._orders:
            return f"ERROR: Order '{order_id}' not found"

        self._orders[order_id]["deadline"] += 3
        return f"OK: Extended {order_id} deadline by 3 (new deadline: {self._orders[order_id]['deadline']})"

    def _escalate_order_impl(self, order_id: str) -> str:
        if order_id not in self._orders:
            return f"ERROR: Order '{order_id}' not found"

        self._orders[order_id]["deadline"] += 5
        return f"OK: Escalated {order_id}. Deadline extended by 5 (new deadline: {self._orders[order_id]['deadline']})"

    def _move_drivers(self) -> int:
        """Move drivers 1 unit along their route. Returns number of deliveries."""
        deliveries = 0
        for d in self._drivers.values():
            if d["status"] == "moving":
                d["eta"] -= 1
                if d["eta"] < 0 and d["route"]:
                    d["location"] = d["route"].pop(0)
                    if not d["route"]:
                        d["status"] = "idle"
                        deliveries += 1
        return deliveries

    def _query_driver_impl(self, driver_id: str) -> str:
        if driver_id not in self._drivers:
            return f"ERROR: Driver '{driver_id}' not found"

        d = self._drivers[driver_id]
        lines = [f"=== DRIVER {driver_id} ==="]
        lines.append(f"Status: {d['status']}")
        lines.append(f"Location: {d['location']}")
        lines.append(f"ETA: {d['eta']}")
        lines.append(
            f"Assigned orders: {', '.join(d['orders']) if d['orders'] else 'None'}"
        )
        if d["route"]:
            lines.append(f"Route: {' -> '.join(d['route'])}")
        return "\n".join(lines)

    def _query_order_impl(self, order_id: str) -> str:
        if order_id not in self._orders:
            return f"ERROR: Order '{order_id}' not found"

        o = self._orders[order_id]
        lines = [f"=== ORDER {order_id} ==="]
        lines.append(f"Status: {o['status']}")
        lines.append(f"Pickup: {o['pickup']}")
        lines.append(f"Dropoff: {o['dropoff']}")
        lines.append(f"Deadline: {o['deadline']}")
        if o["assigned"]:
            lines.append(f"Assigned to: {o['assigned']}")
        if o["delivered"] is not None:
            lines.append(f"Delivered at step: {o['delivered']}")
        return "\n".join(lines)

    def _query_network_impl(self) -> str:
        if not self._network:
            return "ERROR: Network not initialized"

        lines = ["=== NETWORK MAP ==="]
        for node_id in self._network.get_all_nodes():
            node = self._network.get_node(node_id)
            neighbors = self._network.get_neighbors(node_id)
            neighbor_info = []
            for n in neighbors:
                road = self._network.get_road(node_id, n)
                traffic = self._network.get_traffic(node_id, n)
                neighbor_info.append(f"{n}({road.distance}*{traffic:.1f})")
            lines.append(f"{node_id} ({node.node_type}): {', '.join(neighbor_info)}")
        return "\n".join(lines)

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _spawn_drivers(self, n: int) -> Dict[str, Dict[str, Any]]:
        drivers = {}
        for i in range(n):
            drivers[f"D{i}"] = {
                "location": "HUB",
                "route": [],
                "eta": 0,
                "status": "idle",
                "orders": [],
            }
        return drivers

    def _generate_orders(self, n: int) -> Dict[str, Dict[str, Any]]:
        nodes = self._network.get_all_nodes()
        orders = {}
        existing_count = len(self._orders)

        for i in range(n):
            pickup = self._rng.choice(nodes)
            dropoff = self._rng.choice(nodes)
            while dropoff == pickup:
                dropoff = self._rng.choice(nodes)

            oid = f"O{existing_count + i}"
            orders[oid] = {
                "pickup": pickup,
                "dropoff": dropoff,
                "deadline": self._state.time_step + self._rng.randint(10, 30),
                "status": "pending",
                "assigned": None,
                "created": self._state.time_step,
                "delivered": None,
            }
        return orders

    def _update_orders(self):
        """Update order statuses: check deliveries and deadline failures."""
        for oid, o in self._orders.items():
            if o["status"] == "assigned":
                driver = self._drivers[o["assigned"]]
                if driver["location"] == o["dropoff"] and driver["status"] == "idle":
                    o["status"] = "delivered"
                    o["delivered"] = self._state.time_step
            elif o["status"] == "pending" and self._state.time_step > o["deadline"]:
                o["status"] = "failed"
                o["failed_at"] = self._state.time_step

    def _spawn_orders(self):
        """Spawn new orders with 20% probability per step."""
        if self._rng.random() < 0.2:
            new_orders = self._generate_orders(self._rng.randint(1, 2))
            self._orders.update(new_orders)
            for oid in new_orders:
                o = self._orders[oid]
                self._alerts.append(
                    f"New order: {oid}: {o['pickup']} -> {o['dropoff']} (deadline: {o['deadline']})"
                )

    def _update_traffic(self):
        """Randomly trigger traffic spikes."""
        if self._rng.random() < 0.15:
            self._alerts.append("Traffic spike detected")
            self._network.update_traffic(multiplier=1.3)

    def _compute_step_reward(
        self,
        action: "LogiChainAction" = None,
        tool_result: str = "",
        is_error: bool = False,
    ) -> float:
        """Compute reward for the current step."""
        # Terminal rewards - orders that resolved this step
        terminal_reward = self._compute_terminal_reward()

        # Shaping reward - based on action taken
        shaping_reward = self._compute_shaping_reward(action, tool_result, is_error)

        # No progress penalty
        no_progress_penalty = self._check_no_progress_penalty()

        return terminal_reward + shaping_reward + no_progress_penalty

    def _compute_terminal_reward(self) -> float:
        """Compute terminal reward for orders that resolved this step."""
        if not self._reward_config.get("terminal", {}).get("enabled", True):
            return 0.0

        cfg = self._reward_config["terminal"]
        deadline_scaling = cfg.get("deadline_scaling", True)
        reward = 0.0

        for o in self._orders.values():
            if o["status"] == "delivered" and o.get("delivered") == self._state.time_step:
                # Calculate deadline factor
                if deadline_scaling:
                    deadline = o.get("deadline", 30)
                    max_deadline = 30  # Approximate max
                    deadline_factor = 1.0 + (max_deadline - deadline) / max_deadline
                else:
                    deadline_factor = 1.0

                if o["delivered"] <= o["deadline"]:
                    reward += cfg.get("delivery_on_time", 1.0) * deadline_factor
                else:
                    reward += cfg.get("delivery_late", 0.5) * deadline_factor

            elif o["status"] == "failed" and o.get("failed_at") == self._state.time_step:
                reward += cfg.get("order_failed", -0.5)

        return reward

    def _compute_shaping_reward(
        self,
        action: "LogiChainAction" = None,
        tool_result: str = "",
        is_error: bool = False,
    ) -> float:
        """Compute shaping reward based on action taken."""
        cfg = self._reward_config.get("shaping", {})

        if action is None:
            return 0.0

        # Error penalty
        if is_error or (tool_result and tool_result.startswith("ERROR")):
            return cfg.get("error", -0.05)

        tool_name = action.tool_name

        # Query tools - encourage observation
        if tool_name in ("query_driver", "query_order", "query_network"):
            return cfg.get("query", 0.01)

        # Assignment - encourage action
        if tool_name == "assign_order":
            if tool_result and "OK:" in tool_result:
                return cfg.get("assign", 0.02)
            return 0.0

        # Reroute - encourage progress
        if tool_name == "reroute_driver":
            if tool_result and "OK:" in tool_result:
                return cfg.get("reroute", 0.02)
            return 0.0

        # Delay - penalize stalling
        if tool_name == "delay_order":
            return cfg.get("delay", -0.02)

        # Escalate - penalize more heavily
        if tool_name == "escalate_order":
            return cfg.get("escalate", -0.05)

        return 0.0

    def _check_no_progress_penalty(self) -> float:
        """Check for no-progress and apply penalty if threshold exceeded."""
        cfg = self._reward_config.get("no_progress", {})
        threshold = cfg.get("threshold", 10)
        penalty = cfg.get("penalty", -0.05)

        # Check if there's been progress
        current_deliveries = sum(
            1 for o in self._orders.values() if o["status"] == "delivered"
        )
        current_assignments = sum(
            1 for o in self._orders.values() if o["status"] == "assigned"
        )

        has_progress = (
            current_deliveries > self._total_deliveries_at_start
            or current_assignments > self._total_assignments_at_start
        )

        if has_progress:
            self._last_progress_step = self._state.time_step
            self._total_deliveries_at_start = current_deliveries
            self._total_assignments_at_start = current_assignments
            self._steps_without_progress = 0
            return 0.0

        # No progress this step
        self._steps_without_progress += 1

        # Apply penalty if threshold exceeded
        if self._steps_without_progress > threshold:
            return penalty

        return 0.0

    def _is_done(self) -> bool:
        """Check if episode should end based on task conditions."""
        # All orders resolved
        if all(o["status"] in ("delivered", "failed") for o in self._orders.values()):
            return True

        # Safety timeout - prevent infinite loops (500 steps)
        if self._state.time_step >= 500:
            return True

        # No progress timeout - if too many steps without progress
        cfg = self._reward_config.get("no_progress", {})
        max_no_progress = cfg.get("max_no_progress", 50)
        if self._steps_without_progress > max_no_progress:
            return True

        # Task-specific end conditions
        task = self._task_config.get("tasks", {}).get(self._current_task, {})
        done_condition = task.get("done_condition", {})

        if done_condition.get("type") == "time_limit":
            max_time = done_condition.get("max_time_steps", 30)
            if self._state.time_step >= max_time:
                return True

        return False

    def _render_dashboard(self) -> str:
        lines = ["=== DRIVERS ==="]
        for k, d in self._drivers.items():
            lines.append(f"{k}: {d['status']} @ {d['location']}")

        lines.append("\n=== ORDERS ===")
        for k, o in self._orders.items():
            lines.append(f"{k}: {o['status']} deadline={o['deadline']}")

        lines.append(f"\nTime: {self._state.time_step}")

        return "\n".join(lines)
