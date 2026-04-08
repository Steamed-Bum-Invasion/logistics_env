# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Network Graph for Logistics Chain Simulation.

A realistic city-style network with warehouses, delivery zones, and roads.
"""

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Tuple

import networkx as nx


@dataclass
class Road:
    """Represents a road segment between two nodes."""

    name: str
    distance: int  # base travel time in minutes
    capacity: int  # max vehicles before congestion
    road_type: str = "standard"  # highway, arterial, standard


@dataclass
class Node:
    """Represents a location in the network."""

    node_id: str
    node_type: str  # hub, warehouse, zone, customer
    name: str = ""
    x: float = 0.0
    y: float = 0.0


class NetworkGraph:
    """
    Realistic logistics network graph with bidirectional edges,
    road types, traffic simulation, and pre-computed shortest paths.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = Random(seed)

        self.nodes: Dict[str, Node] = {}
        self.roads: Dict[Tuple[str, str], Road] = {}
        self.traffic: Dict[Tuple[str, str], float] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._all_pairs_paths: Dict[Tuple[str, str], Tuple[int, List[str]]] = {}

        self._build_network()
        self._generate_traffic()
        self._precompute_all_pairs()

    def _build_network(self) -> None:
        nodes = [
            Node("HUB", "hub", "Central Distribution Hub", 0, 0),
            Node("W1", "warehouse", "Warehouse Alpha", -3, 2),
            Node("W2", "warehouse", "Warehouse Beta", 3, 2),
            Node("Z1", "zone", "North District", -2, 4),
            Node("Z2", "zone", "East District", 2, 4),
            Node("Z3", "zone", "South District", 0, -3),
            Node("Z4", "zone", "West District", -4, 0),
            Node("Z5", "zone", "Central Business", 1, 1),
            Node("C1", "customer", "Customer Zone A", -1, 5),
            Node("C2", "customer", "Customer Zone B", 4, 3),
            Node("C3", "customer", "Customer Zone C", 0, -5),
            Node("C4", "customer", "Customer Zone D", -5, -1),
        ]

        for node in nodes:
            self.nodes[node.node_id] = node
            self._graph.add_node(
                node.node_id,
                node_type=node.node_type,
                name=node.name,
            )

        edges = [
            ("HUB", "W1", 8, 20, "arterial"),
            ("HUB", "W2", 10, 25, "arterial"),
            ("HUB", "Z5", 5, 30, "standard"),
            ("HUB", "Z3", 7, 20, "arterial"),
            ("W1", "Z1", 6, 15, "standard"),
            ("W1", "Z4", 9, 15, "standard"),
            ("W1", "Z5", 7, 20, "arterial"),
            ("W2", "Z2", 6, 15, "standard"),
            ("W2", "Z5", 8, 20, "arterial"),
            ("W2", "C2", 10, 10, "standard"),
            ("Z1", "C1", 5, 10, "standard"),
            ("Z2", "C2", 7, 10, "standard"),
            ("Z3", "C3", 5, 10, "standard"),
            ("Z4", "C4", 4, 10, "standard"),
            ("Z5", "Z1", 4, 25, "arterial"),
            ("Z5", "Z2", 4, 25, "arterial"),
            ("Z5", "Z3", 6, 20, "standard"),
            ("Z5", "Z4", 5, 20, "standard"),
        ]

        road_names = [
            "Main St",
            "Oak Ave",
            "Elm Blvd",
            "Park Dr",
            "Commerce Way",
            "Industrial Rd",
            "Market St",
            "Harbor View",
            "Center St",
            "North Ave",
        ]

        for i, (u, v, dist, cap, rtype) in enumerate(edges):
            name = road_names[i % len(road_names)]
            self.roads[(u, v)] = Road(name, dist, cap, rtype)
            self.roads[(v, u)] = Road(name, dist, cap, rtype)

            self._graph.add_edge(u, v, distance=dist, road_type=rtype, capacity=cap)
            self._graph.add_edge(v, u, distance=dist, road_type=rtype, capacity=cap)

    def _generate_traffic(self) -> None:
        for (u, v), road in self.roads.items():
            if road.road_type == "highway":
                base = self._rng.choice([0.8, 0.9, 1.0, 1.1])
            elif road.road_type == "arterial":
                base = self._rng.choice([0.9, 1.0, 1.0, 1.2, 1.5])
            else:
                base = self._rng.choice([1.0, 1.0, 1.2, 1.5, 2.0])
            self.traffic[(u, v)] = base

    def _precompute_all_pairs(self) -> None:
        for node_a in self.nodes:
            for node_b in self.nodes:
                if node_a != node_b:
                    cost, path = self.shortest_path(node_a, node_b)
                    self._all_pairs_paths[(node_a, node_b)] = (cost, path)

    def shortest_path(self, start: str, end: str) -> Tuple[int, List[str]]:
        if start == end:
            return 0, [start]

        cached = self._all_pairs_paths.get((start, end))
        if cached is not None:
            return cached

        try:
            path = nx.dijkstra_path(self._graph, start, end, weight=self._edge_weight)
            cost = sum(self._edge_weight(u, v) for u, v in zip(path[:-1], path[1:]))
            return int(cost), path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 9999, []

    def _edge_weight(self, u: str, v: str, edge_data: Optional[Dict] = None) -> float:
        base = self._graph[u][v].get("distance", 1)
        traffic = self.traffic.get((u, v), 1.0)
        return base * traffic

    def update_traffic(
        self, node: Optional[str] = None, multiplier: float = 1.5
    ) -> None:
        if node is None:
            for key in self.traffic:
                self.traffic[key] *= multiplier
        else:
            for u, v in self.roads:
                if u == node:
                    self.traffic[(u, v)] *= multiplier
        
        self._all_pairs_paths.clear()
        self._precompute_all_pairs()

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_road(self, from_node: str, to_node: str) -> Optional[Road]:
        return self.roads.get((from_node, to_node))

    def get_traffic(self, from_node: str, to_node: str) -> float:
        return self.traffic.get((from_node, to_node), 1.0)

    def get_all_nodes(self) -> List[str]:
        return list(self.nodes.keys())

    def get_neighbors(self, node: str) -> List[str]:
        return list(self._graph.neighbors(node))

    def render_network(self) -> str:
        lines = ["=== NETWORK MAP ==="]
        for node_id, node in self.nodes.items():
            neighbors = self.get_neighbors(node_id)
            lines.append(f"{node_id} ({node.node_type}): {', '.join(neighbors)}")
        return "\n".join(lines)
