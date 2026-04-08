# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Network Graph for Logistics Chain Simulation.

A realistic city-style network with warehouses, delivery zones, and roads.
Supports procedural generation and configuration-driven networks.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import yaml


@dataclass
class Road:
    """Represents a road segment between two nodes."""

    name: str
    distance: int
    capacity: int
    road_type: str = "standard"


@dataclass
class Node:
    """Represents a location in the network."""

    node_id: str
    node_type: str
    name: str = ""
    x: float = 0.0
    y: float = 0.0


@dataclass
class NetworkConfig:
    """Configuration for procedural network generation."""

    num_hubs: int = 1
    num_warehouses: int = 2
    num_zones: int = 5
    num_customers: int = 4
    connectivity: float = 0.3
    max_edge_distance: int = 12
    variation_mode: str = "fixed"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkConfig":
        return cls(
            num_hubs=data.get("num_hubs", 1),
            num_warehouses=data.get("num_warehouses", 2),
            num_zones=data.get("num_zones", 5),
            num_customers=data.get("num_customers", 4),
            connectivity=data.get("connectivity", 0.3),
            max_edge_distance=data.get("max_edge_distance", 12),
            variation_mode=data.get("variation_mode", "fixed"),
        )


ROAD_NAMES = [
    "Main St", "Oak Ave", "Elm Blvd", "Park Dr", "Commerce Way",
    "Industrial Rd", "Market St", "Harbor View", "Center St", "North Ave",
    "South Blvd", "West End", "East Side", "Central Ave", "Transit Way",
    "Logistics Ln", "Depot Rd", "Terminal Ave", "Hub Circle", "Zone Pkwy",
]

NODE_NAMES = {
    "hub": ["Central Hub", "Main Hub", "Distribution Center", "Logistics Hub"],
    "warehouse": ["Warehouse Alpha", "Warehouse Beta", "Warehouse Gamma", "Warehouse Delta", "Warehouse Epsilon"],
    "zone": ["North District", "South District", "East District", "West District", "Central District", "Uptown", "Downtown", "Midtown", "Suburbia", "Industrial Zone"],
    "customer": ["Customer Zone A", "Customer Zone B", "Customer Zone C", "Customer Zone D", "Customer Zone E", "Customer Zone F", "Customer Zone G", "Customer Zone H"],
}


class NetworkGraph:
    """
    Realistic logistics network graph with bidirectional edges,
    road types, traffic simulation, and pre-computed shortest paths.

    Supports:
    - Default hardcoded network (backward compatible)
    - Procedural generation from parameters
    - Configuration-driven networks from YAML/dict
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = Random(seed)
        self._seed = seed

        self.nodes: Dict[str, Node] = {}
        self.roads: Dict[Tuple[str, str], Road] = {}
        self.traffic: Dict[Tuple[str, str], float] = {}
        self._graph: nx.DiGraph = nx.DiGraph()
        self._all_pairs_paths: Dict[Tuple[str, str], Tuple[int, List[str]]] = {}

        self._build_default_network()
        self._generate_traffic()
        self._precompute_all_pairs()

    @classmethod
    def from_config(cls, config_path: str, seed: Optional[int] = None) -> "NetworkGraph":
        """
        Load network from YAML configuration file.

        Args:
            config_path: Path to YAML config file
            seed: Random seed for traffic generation

        Returns:
            NetworkGraph instance
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config, seed=seed)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], seed: Optional[int] = None) -> "NetworkGraph":
        """
        Create network from dictionary configuration.

        Supports two formats:
        1. Procedural: {num_hubs, num_warehouses, num_zones, num_customers, ...}
        2. Explicit: {nodes: [...], edges: [...]}

        Args:
            config: Configuration dictionary
            seed: Random seed

        Returns:
            NetworkGraph instance
        """
        instance = cls.__new__(cls)
        instance._rng = Random(seed)
        instance._seed = seed
        instance.nodes = {}
        instance.roads = {}
        instance.traffic = {}
        instance._graph = nx.DiGraph()
        instance._all_pairs_paths = {}

        if "nodes" in config and "edges" in config:
            instance._build_explicit_network(config["nodes"], config["edges"])
        else:
            net_config = NetworkConfig.from_dict(config)
            instance._build_procedural_network(net_config)

        instance._generate_traffic()
        instance._precompute_all_pairs()
        return instance

    @classmethod
    def generate(
        cls,
        seed: Optional[int] = None,
        num_hubs: int = 1,
        num_warehouses: int = 2,
        num_zones: int = 5,
        num_customers: int = 4,
        connectivity: float = 0.3,
        max_edge_distance: int = 12,
    ) -> "NetworkGraph":
        """
        Generate a hierarchical logistics network.

        Args:
            seed: Random seed for reproducibility
            num_hubs: Number of hub nodes
            num_warehouses: Number of warehouses
            num_zones: Number of delivery zones
            num_customers: Number of customer areas
            connectivity: Edge density factor (0.0-1.0)
            max_edge_distance: Maximum road distance

        Returns:
            NetworkGraph instance with generated topology
        """
        return cls.from_dict({
            "num_hubs": num_hubs,
            "num_warehouses": num_warehouses,
            "num_zones": num_zones,
            "num_customers": num_customers,
            "connectivity": connectivity,
            "max_edge_distance": max_edge_distance,
        }, seed=seed)

    def _build_default_network(self) -> None:
        """Build the default hardcoded network (backward compatible)."""
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

        self._add_nodes(nodes)
        self._add_edges(edges)

    def _build_explicit_network(self, nodes_data: List[Dict], edges_data: List[Dict]) -> None:
        """Build network from explicit node and edge definitions."""
        nodes = []
        for n in nodes_data:
            nodes.append(Node(
                node_id=n["id"],
                node_type=n.get("type", "zone"),
                name=n.get("name", n["id"]),
                x=n.get("x", 0.0),
                y=n.get("y", 0.0),
            ))

        edges = []
        for e in edges_data:
            edges.append((
                e["from"],
                e["to"],
                e.get("distance", 5),
                e.get("capacity", 15),
                e.get("road_type", "standard"),
            ))

        self._add_nodes(nodes)
        self._add_edges(edges)

    def _build_procedural_network(self, config: NetworkConfig) -> None:
        """Generate network procedurally from configuration."""
        nodes = []
        node_id_counter = {"hub": 0, "warehouse": 0, "zone": 0, "customer": 0}

        def get_node_id(node_type: str) -> str:
            if node_type == "hub":
                node_id_counter["hub"] += 1
                return "HUB" if node_id_counter["hub"] == 1 else f"H{node_id_counter['hub']}"
            elif node_type == "warehouse":
                node_id_counter["warehouse"] += 1
                return f"W{node_id_counter['warehouse']}"
            elif node_type == "zone":
                node_id_counter["zone"] += 1
                return f"Z{node_id_counter['zone']}"
            else:
                node_id_counter["customer"] += 1
                return f"C{node_id_counter['customer']}"

        def get_node_name(node_type: str, idx: int) -> str:
            names = NODE_NAMES.get(node_type, [node_type.title()])
            return names[(idx - 1) % len(names)]

        hubs = []
        for i in range(1, config.num_hubs + 1):
            angle = 2 * math.pi * (i - 1) / max(config.num_hubs, 1)
            x = 2 * math.cos(angle)
            y = 2 * math.sin(angle)
            node_id = get_node_id("hub")
            node = Node(node_id, "hub", get_node_name("hub", i), x, y)
            nodes.append(node)
            hubs.append(node_id)

        warehouses = []
        for i in range(1, config.num_warehouses + 1):
            angle = 2 * math.pi * (i - 1) / max(config.num_warehouses, 1) + 0.3
            radius = 4 + self._rng.random() * 2
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node_id = get_node_id("warehouse")
            node = Node(node_id, "warehouse", get_node_name("warehouse", i), x, y)
            nodes.append(node)
            warehouses.append(node_id)

        zones = []
        for i in range(1, config.num_zones + 1):
            angle = 2 * math.pi * (i - 1) / max(config.num_zones, 1) + 0.5
            radius = 7 + self._rng.random() * 3
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node_id = get_node_id("zone")
            node = Node(node_id, "zone", get_node_name("zone", i), x, y)
            nodes.append(node)
            zones.append(node_id)

        customers = []
        for i in range(1, config.num_customers + 1):
            angle = 2 * math.pi * (i - 1) / max(config.num_customers, 1) + 0.7
            radius = 10 + self._rng.random() * 4
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node_id = get_node_id("customer")
            node = Node(node_id, "customer", get_node_name("customer", i), x, y)
            nodes.append(node)
            customers.append(node_id)

        self._add_nodes(nodes)

        edges = []
        road_idx = 0

        def add_edge(u: str, v: str, road_type: str = "standard"):
            nonlocal road_idx
            dist = self._compute_distance(u, v)
            dist = min(dist, config.max_edge_distance)
            cap = {"highway": 40, "arterial": 25, "standard": 15}.get(road_type, 15)
            edges.append((u, v, dist, cap, road_type))
            road_idx += 1

        for hub in hubs:
            for wh in warehouses:
                add_edge(hub, wh, "arterial")

        for wh in warehouses:
            k = max(1, int(len(zones) * config.connectivity))
            nearest_zones = self._find_nearest(wh, zones, k)
            for z in nearest_zones:
                add_edge(wh, z, "standard")

        for z in zones:
            k = max(1, int(len(customers) * config.connectivity))
            nearest_customers = self._find_nearest(z, customers, k)
            for c in nearest_customers:
                add_edge(z, c, "standard")

        for i, z1 in enumerate(zones):
            for z2 in zones[i + 1:]:
                if self._rng.random() < config.connectivity * 0.5:
                    add_edge(z1, z2, "arterial")

        for i, h1 in enumerate(hubs):
            for h2 in hubs[i + 1:]:
                add_edge(h1, h2, "highway")

        self._add_edges(edges)

    def _add_nodes(self, nodes: List[Node]) -> None:
        """Add nodes to the graph."""
        for node in nodes:
            self.nodes[node.node_id] = node
            self._graph.add_node(
                node.node_id,
                node_type=node.node_type,
                name=node.name,
            )

    def _add_edges(self, edges: List[Tuple]) -> None:
        """Add bidirectional edges to the graph."""
        for i, (u, v, dist, cap, rtype) in enumerate(edges):
            name = ROAD_NAMES[i % len(ROAD_NAMES)]
            self.roads[(u, v)] = Road(name, dist, cap, rtype)
            self.roads[(v, u)] = Road(name, dist, cap, rtype)

            self._graph.add_edge(u, v, distance=dist, road_type=rtype, capacity=cap)
            self._graph.add_edge(v, u, distance=dist, road_type=rtype, capacity=cap)

    def _compute_distance(self, node_a: str, node_b: str) -> int:
        """Compute Euclidean distance between two nodes."""
        a = self.nodes[node_a]
        b = self.nodes[node_b]
        dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        return max(1, int(dist))

    def _find_nearest(self, source: str, targets: List[str], k: int) -> List[str]:
        """Find k nearest nodes from source to targets."""
        distances = [(t, self._compute_distance(source, t)) for t in targets]
        distances.sort(key=lambda x: x[1])
        return [t for t, _ in distances[:k]]

    def _generate_traffic(self) -> None:
        """Generate initial traffic multipliers for all roads."""
        for (u, v), road in self.roads.items():
            if road.road_type == "highway":
                base = self._rng.choice([0.8, 0.9, 1.0, 1.1])
            elif road.road_type == "arterial":
                base = self._rng.choice([0.9, 1.0, 1.0, 1.2, 1.5])
            else:
                base = self._rng.choice([1.0, 1.0, 1.2, 1.5, 2.0])
            self.traffic[(u, v)] = base

    def _precompute_all_pairs(self) -> None:
        """Precompute shortest paths between all node pairs."""
        for node_a in self.nodes:
            for node_b in self.nodes:
                if node_a != node_b:
                    cost, path = self._compute_shortest_path(node_a, node_b)
                    self._all_pairs_paths[(node_a, node_b)] = (cost, path)

    def _compute_shortest_path(self, start: str, end: str) -> Tuple[int, List[str]]:
        """Compute shortest path using Dijkstra's algorithm."""
        if start == end:
            return 0, [start]

        try:
            path = nx.dijkstra_path(self._graph, start, end, weight=self._edge_weight)
            cost = sum(self._edge_weight(u, v) for u, v in zip(path[:-1], path[1:]))
            return int(cost), path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 9999, []

    def shortest_path(self, start: str, end: str) -> Tuple[int, List[str]]:
        """Get shortest path between two nodes (cached)."""
        if start == end:
            return 0, [start]

        cached = self._all_pairs_paths.get((start, end))
        if cached is not None:
            return cached

        return self._compute_shortest_path(start, end)

    def _edge_weight(self, u: str, v: str, edge_data: Optional[Dict] = None) -> float:
        """Compute edge weight considering traffic."""
        base = self._graph[u][v].get("distance", 1)
        traffic = self.traffic.get((u, v), 1.0)
        return base * traffic

    def update_traffic(
        self, node: Optional[str] = None, multiplier: float = 1.5
    ) -> None:
        """Update traffic multipliers (global or node-local)."""
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
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_road(self, from_node: str, to_node: str) -> Optional[Road]:
        """Get road between two nodes."""
        return self.roads.get((from_node, to_node))

    def get_traffic(self, from_node: str, to_node: str) -> float:
        """Get traffic multiplier for a road."""
        return self.traffic.get((from_node, to_node), 1.0)

    def get_all_nodes(self) -> List[str]:
        """Get all node IDs."""
        return list(self.nodes.keys())

    def get_neighbors(self, node: str) -> List[str]:
        """Get neighbors of a node."""
        return list(self._graph.neighbors(node))

    def get_hub_nodes(self) -> List[str]:
        """Get all hub node IDs."""
        return [nid for nid, n in self.nodes.items() if n.node_type == "hub"]

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """Get all node IDs of a specific type."""
        return [nid for nid, n in self.nodes.items() if n.node_type == node_type]

    def render_network(self) -> str:
        """Render network as text."""
        lines = ["=== NETWORK MAP ==="]
        for node_id, node in self.nodes.items():
            neighbors = self.get_neighbors(node_id)
            lines.append(f"{node_id} ({node.node_type}): {', '.join(neighbors)}")
        return "\n".join(lines)
