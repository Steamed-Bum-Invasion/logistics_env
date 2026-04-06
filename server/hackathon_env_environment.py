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

import uuid
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
)
from openenv.core.env_server.types import Observation

try:
    from ..models import (
        AVAILABLE_TOOLS,
        LogiChainAction,
        LogiChainState,
        LogiChainObservation,
        LogiChainToolObservation,
    )
    from .network_graph import NetworkGraph
except ImportError:
    from models import (
        AVAILABLE_TOOLS,
        LogiChainAction,
        LogiChainState,
        LogiChainObservation,
        LogiChainToolObservation,
    )
    from server.network_graph import NetworkGraph


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
    New orders arrive each shift. The agent must assign drivers, route them,
    and handle delays/escalations.

    Tools:
        assign_order: Assign a pending order to a driver
        reroute_driver: Reroute a driver to their order's dropoff
        delay_order: Extend an order's deadline by 3 steps
        escalate_order: Escalate an order (extend deadline by 5, costs more)
        advance_shift: Advance time, process deliveries, spawn new orders
        query_driver: Get driver status and location
        query_order: Get order status and details
        query_network: Get network topology and traffic info
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._network: Optional[NetworkGraph] = None
        self._drivers: Dict[str, Dict[str, Any]] = {}
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[str] = []
        self._step_count = 0
        self._actions_remaining = 3
        self._rng = __import__("random").Random()
        self._state = LogiChainState()

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
        def advance_shift() -> str:
            """
            Advance the simulation by one shift.

            Processes driver movement, deliveries, and spawns new orders.
            Resets your action budget for the next shift.
            Unspent actions are lost — no rollover.

            Returns:
                Shift summary report with delivery results and new orders
            """
            return self._advance_shift_impl()

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
        self._actions_remaining = 3

        self._network = NetworkGraph(seed=seed)
        self._drivers = self._spawn_drivers(5)
        self._orders = self._generate_orders(10)

        ep_id = episode_id or str(uuid.uuid4())

        return LogiChainObservation(
            dashboard_text=self._render_dashboard(),
            alerts=list(self._alerts),
            available_tools=list(AVAILABLE_TOOLS),
            episode_id=ep_id,
            done=False,
            reward=0.0,
            remaining_actions=self._actions_remaining,
        )

    def step(
        self,
        action: LogiChainAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._step_count += 1

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
            remaining_actions=self._actions_remaining,
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
        is_error = (call_obs.error is not None) or result_text.startswith("ERROR")

        if action.tool_name == "advance_shift":
            reward = self._compute_shift_reward()
            done = self._is_done()
        else:
            reward = self._shaping_reward(action.tool_name, result_text, is_error)
            done = False

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
            remaining_actions=self._actions_remaining,
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
            episode_id=getattr(self, "_episode_id", ""),
            step_count=self._step_count,
            time_step=self._state.time_step,
            max_steps=self._state.max_steps,
            actions_remaining=self._actions_remaining,
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

        driver["route"] = path
        driver["eta"] = cost
        driver["status"] = "moving"
        driver["orders"].append(order_id)

        order["status"] = "assigned"
        order["assigned"] = driver_id

        self._actions_remaining -= 1
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
        driver["route"] = path
        driver["eta"] = cost

        self._actions_remaining -= 1
        return f"OK: Rerouted {driver_id} to {order['dropoff']}. Route: {' -> '.join(path)}. ETA: {cost}"

    def _delay_order_impl(self, order_id: str) -> str:
        if order_id not in self._orders:
            return f"ERROR: Order '{order_id}' not found"

        self._orders[order_id]["deadline"] += 3
        self._actions_remaining -= 1
        return f"OK: Extended {order_id} deadline by 3 (new deadline: {self._orders[order_id]['deadline']})"

    def _escalate_order_impl(self, order_id: str) -> str:
        if order_id not in self._orders:
            return f"ERROR: Order '{order_id}' not found"

        self._orders[order_id]["deadline"] += 5
        self._actions_remaining -= 1
        return f"OK: Escalated {order_id}. Deadline extended by 5 (new deadline: {self._orders[order_id]['deadline']})"

    def _advance_shift_impl(self) -> str:
        self._state.time_step += 1
        self._alerts = []

        deliveries = self._update_drivers()
        self._update_orders()

        if self._rng.random() < 0.15:
            self._alerts.append("Traffic spike detected")
            self._network.update_traffic(multiplier=1.3)

        new_orders = self._generate_orders(self._rng.randint(1, 3))
        self._orders.update(new_orders)

        self._actions_remaining = 3

        lines = [f"=== SHIFT {self._state.time_step} SUMMARY ==="]
        lines.append(f"Deliveries completed: {deliveries}")
        lines.append(f"New orders received: {len(new_orders)}")
        for oid in new_orders:
            o = self._orders[oid]
            lines.append(
                f"  {oid}: {o['pickup']} -> {o['dropoff']} (deadline: {o['deadline']})"
            )
        if self._alerts:
            lines.append(f"\nAlerts: {', '.join(self._alerts)}")
        lines.append(f"\nActions remaining: {self._actions_remaining}")

        return "\n".join(lines)

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

    def _update_drivers(self) -> int:
        deliveries = 0
        for d in self._drivers.values():
            if d["status"] == "moving":
                d["eta"] -= 1
                if d["eta"] <= 0 and d["route"]:
                    d["location"] = d["route"].pop(0)
                    if not d["route"]:
                        d["status"] = "idle"
                        deliveries += 1
        return deliveries

    def _update_orders(self):
        for oid, o in self._orders.items():
            if o["status"] == "assigned":
                driver = self._drivers[o["assigned"]]
                if driver["location"] == o["dropoff"] and driver["status"] == "idle":
                    o["status"] = "delivered"
                    o["delivered"] = self._state.time_step
            elif o["status"] == "pending" and self._state.time_step > o["deadline"]:
                o["status"] = "failed"

    def _compute_shift_reward(self) -> float:
        reward = 0.0
        for o in self._orders.values():
            if o["status"] == "delivered" and o["delivered"] == self._state.time_step:
                if o["delivered"] <= o["deadline"]:
                    reward += 1.0
                else:
                    reward += 0.5
            elif o["status"] == "failed":
                reward -= 0.5
        return reward

    def _shaping_reward(
        self, tool_name: str, result_text: str, is_error: bool
    ) -> float:
        if is_error:
            return -0.05

        if tool_name in ("query_driver", "query_order", "query_network"):
            return 0.01

        if tool_name in ("assign_order", "reroute_driver") and not is_error:
            return 0.02

        if tool_name == "delay_order":
            return -0.02

        if tool_name == "escalate_order":
            return -0.05

        return 0.0

    def _is_done(self) -> bool:
        if self._state.time_step >= self._state.max_steps:
            return True

        if all(o["status"] in ("delivered", "failed") for o in self._orders.values()):
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
        lines.append(f"Actions remaining: {self._actions_remaining}")

        return "\n".join(lines)
