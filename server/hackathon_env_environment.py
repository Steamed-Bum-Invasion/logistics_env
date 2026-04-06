# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hackathon Env Environment Implementation.

"""

from random import Random, choice
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from .network_graph import NetworkGraph

try:
    from ..models import LogiChainAction, LogiChainObservation, LogiChainState
except ImportError:
    from models import LogiChainAction, LogiChainObservation, LogiChainState


class LogiChainEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = LogiChainState()
        self._alerts: List[str] = []
        self._last_reward = 0.0
        self._network: Optional[NetworkGraph] = None
        # HACK: for some reason i wasnt getting random.seed(), so used Random as a hack
        self._rng = Random()

    # ---------------------------
    # RESET
    # ---------------------------
    def reset(self, seed=None, episode_id=None, **kwargs) -> LogiChainObservation:
        if seed is not None:
            self._rng.seed(seed)

        self._alerts = []

        self._network = NetworkGraph(seed=seed)

        self._state = LogiChainState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            time_step=0,
            max_steps=50,
            graph=self._serialize_graph(),
            drivers=self._spawn_drivers(5),
            orders=self._generate_orders(10),
            traffic=self._serialize_traffic(),
            metrics={"completed": 0, "failed": 0, "on_time": 0},
        )

        return self._build_observation(remaining_actions=self._get_action_budget())

    def _serialize_graph(self) -> Dict[str, Dict[str, int]]:
        serialized = {}
        for node_id in self._network.get_all_nodes():
            serialized[node_id] = {}
            for neighbor in self._network.get_neighbors(node_id):
                road = self._network.get_road(node_id, neighbor)
                if road:
                    serialized[node_id][neighbor] = road.distance
        return serialized

    def _serialize_traffic(self) -> Dict[str, float]:
        return {f"{k[0]}->{k[1]}": v for k, v in self._network.traffic.items()}

    # ---------------------------
    # STEP
    # ---------------------------
    def step(self, action: LogiChainAction, **kwargs) -> LogiChainObservation:
        self._alerts = []
        reward = 0.0

        budget = self._get_action_budget()
        actions = action.actions[:budget]

        if len(action.actions) > budget:
            reward -= 0.2 * (len(action.actions) - budget)

        for a in actions:
            valid, r = self._apply_atomic_action(a)
            reward += r

        reward -= 0.02 * len(actions)

        self._advance_time()
        self._update_drivers()
        self._update_orders()

        self._inject_events()

        reward += self._compute_delivery_rewards()

        done = self._is_done()

        self._last_reward = reward

        return LogiChainObservation(
            done=done,
            reward=reward,
            dashboard_text=self._render_dashboard(),
            alerts=self._alerts,
            available_actions=self._available_actions(),
            remaining_actions=self._get_action_budget(),
        )

    # ---------------------------
    # STATE
    # ---------------------------
    @property
    def state(self) -> LogiChainState:
        return self._state

    # ===========================
    # DRIVERS
    # ===========================

    def _spawn_drivers(self, n) -> Dict[str, Dict[str, Any]]:
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

    # ===========================
    # ORDERS
    # ===========================

    def _generate_orders(self, n) -> Dict[str, Dict[str, Any]]:
        nodes = self._network.get_all_nodes()
        orders = {}

        for i in range(n):
            pickup = choice(nodes)
            dropoff = choice(nodes)
            while dropoff == pickup:
                dropoff = choice(nodes)

            orders[f"O{i}"] = {
                "pickup": pickup,
                "dropoff": dropoff,
                "deadline": self._rng.randint(10, 30),
                "status": "pending",
                "assigned": None,
                "created": 0,
                "delivered": None,
            }
        return orders

    # ===========================
    # ACTIONS (TOOLS)
    # ===========================

    def _apply_atomic_action(self, a: Dict[str, Any]):
        t = a.get("action_type")

        if t == "assign_order":
            return True, self._assign_order(a)
        elif t == "reroute_driver":
            return True, self._reroute_driver(a)
        elif t == "delay_order":
            return True, self._delay_order(a)
        elif t == "escalate_order":
            return True, self._escalate_order(a)
        else:
            self._alerts.append(f"Invalid action: {t}")
            return False, -0.1

    def _assign_order(self, a):
        oid = a.get("order_id")
        did = a.get("driver_id")

        if oid not in self._state.orders or did not in self._state.drivers:
            return -0.1

        order = self._state.orders[oid]
        driver = self._state.drivers[did]

        if order["status"] != "pending":
            return -0.05

        cost, path = self._network.shortest_path(driver["location"], order["pickup"])

        driver["route"] = path
        driver["eta"] = cost
        driver["status"] = "moving"
        driver["orders"].append(oid)

        order["status"] = "assigned"
        order["assigned"] = did

        return 0.0

    def _reroute_driver(self, a):
        did = a.get("driver_id")
        if did not in self._state.drivers:
            return -0.1

        driver = self._state.drivers[did]
        if not driver["orders"]:
            return -0.05

        oid = driver["orders"][0]
        order = self._state.orders[oid]

        cost, path = self._network.shortest_path(driver["location"], order["dropoff"])
        driver["route"] = path
        driver["eta"] = cost

        return 0.0

    def _delay_order(self, a):
        oid = a.get("order_id")
        if oid not in self._state.orders:
            return -0.1

        self._state.orders[oid]["deadline"] += 3
        return -0.02

    def _escalate_order(self, a):
        oid = a.get("order_id")
        if oid not in self._state.orders:
            return -0.1

        self._state.orders[oid]["deadline"] += 5
        return -0.1

    # ===========================
    # SIMULATION
    # ===========================

    def _advance_time(self):
        self._state.time_step += 1
        self._state.step_count += 1

    def _update_drivers(self):
        for d in self._state.drivers.values():
            if d["status"] == "moving":
                d["eta"] -= 1
                if d["eta"] <= 0 and d["route"]:
                    d["location"] = d["route"].pop(0)
                    if not d["route"]:
                        d["status"] = "idle"

    def _update_orders(self):
        for oid, o in self._state.orders.items():
            if o["status"] == "assigned":
                d = self._state.drivers[o["assigned"]]

                if d["location"] == o["dropoff"]:
                    o["status"] = "delivered"
                    o["delivered"] = self._state.time_step

    def _inject_events(self):
        if self._rng.random() < 0.1:
            self._alerts.append("Traffic spike detected")
            if self._network:
                self._network.update_traffic(multiplier=1.3)

    # ===========================
    # REWARD
    # ===========================

    def _compute_delivery_rewards(self):
        r = 0.0
        for o in self._state.orders.values():
            if o["status"] == "delivered" and o["delivered"] == self._state.time_step:
                if o["delivered"] <= o["deadline"]:
                    r += 1.0
                else:
                    r += 0.5
        return r

    # ===========================
    # OBSERVATION
    # ===========================

    def _render_dashboard(self):
        lines = ["=== DRIVERS ==="]
        for k, d in self._state.drivers.items():
            lines.append(f"{k}: {d['status']} @ {d['location']}")

        lines.append("\n=== ORDERS ===")
        for k, o in self._state.orders.items():
            lines.append(f"{k}: {o['status']} deadline={o['deadline']}")

        lines.append(f"\nTime: {self._state.time_step}")

        return "\n".join(lines)

    def _build_observation(self, remaining_actions):
        return LogiChainObservation(
            done=False,
            reward=None,
            dashboard_text=self._render_dashboard(),
            alerts=self._alerts,
            available_actions=self._available_actions(),
            remaining_actions=remaining_actions,
        )

    def _available_actions(self):
        return [
            "assign_order",
            "reroute_driver",
            "delay_order",
            "escalate_order",
            "noop",
        ]

    def _get_action_budget(self):
        return 3

    def _is_done(self):
        if self._state.time_step >= self._state.max_steps:
            return True

        if all(
            o["status"] in ["delivered", "failed"] for o in self._state.orders.values()
        ):
            return True

        return False

