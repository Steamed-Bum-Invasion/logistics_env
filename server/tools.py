# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tool implementations for LogiChain environment.

Each tool is a standalone function that operates on the environment state.
Tools are called by the agent via MCP (FastMCP) tool dispatch.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logistics_env.server.logistics_environment import LogiChainEnvironment


def assign_order(env: "LogiChainEnvironment", order_id: str, driver_id: str) -> str:
    """
    Assign a pending order to a driver.

    Args:
        env: The environment instance
        order_id: Order identifier (e.g. 'O0', 'O1')
        driver_id: Driver identifier (e.g. 'D0', 'D1')

    Returns:
        Confirmation with route info, or error if order/driver invalid
    """
    if order_id not in env._orders:
        return f"ERROR: Order '{order_id}' not found"
    if driver_id not in env._drivers:
        return f"ERROR: Driver '{driver_id}' not found"

    order = env._orders[order_id]
    driver = env._drivers[driver_id]

    if order["status"] != "pending":
        return f"ERROR: Order '{order_id}' is not pending (status: {order['status']})"

    cost, path = env._network.shortest_path(driver["location"], order["pickup"])

    driver["route"] = path[1:] if path and path[0] == driver["location"] else path
    driver["eta"] = cost
    driver["status"] = "moving"
    driver["orders"].append(order_id)

    order["status"] = "assigned"
    order["assigned"] = driver_id

    return f"OK: Assigned {order_id} to {driver_id}. Route: {' -> '.join(path)}. ETA: {cost}"


def reroute_driver(env: "LogiChainEnvironment", driver_id: str) -> str:
    """
    Reroute a driver to their assigned order's dropoff location.

    Args:
        env: The environment instance
        driver_id: Driver identifier

    Returns:
        Updated route info, or error if driver has no orders
    """
    if driver_id not in env._drivers:
        return f"ERROR: Driver '{driver_id}' not found"

    driver = env._drivers[driver_id]
    if not driver["orders"]:
        return f"ERROR: Driver '{driver_id}' has no assigned orders"

    oid = driver["orders"][0]
    order = env._orders[oid]

    cost, path = env._network.shortest_path(driver["location"], order["dropoff"])
    driver["route"] = path[1:] if path and path[0] == driver["location"] else path
    driver["eta"] = cost
    driver["status"] = "moving"

    return f"OK: Rerouted {driver_id} to {order['dropoff']}. Route: {' -> '.join(path)}. ETA: {cost}"


def delay_order(env: "LogiChainEnvironment", order_id: str) -> str:
    """
    Extend an order's deadline by 3 steps.

    Args:
        env: The environment instance
        order_id: Order identifier

    Returns:
        Confirmation with new deadline, or error if order invalid
    """
    if order_id not in env._orders:
        return f"ERROR: Order '{order_id}' not found"

    env._orders[order_id]["deadline"] += 3
    return f"OK: Extended {order_id} deadline by 3 (new deadline: {env._orders[order_id]['deadline']})"


def escalate_order(env: "LogiChainEnvironment", order_id: str) -> str:
    """
    Escalate an order with priority handling (extend deadline by 5 steps).

    Args:
        env: The environment instance
        order_id: Order identifier

    Returns:
        Confirmation with new deadline, or error if order invalid
    """
    if order_id not in env._orders:
        return f"ERROR: Order '{order_id}' not found"

    env._orders[order_id]["deadline"] += 5
    return f"OK: Escalated {order_id}. Deadline extended by 5 (new deadline: {env._orders[order_id]['deadline']})"


def query_driver(env: "LogiChainEnvironment", driver_id: str) -> str:
    """
    Get detailed status of a driver.

    Args:
        env: The environment instance
        driver_id: Driver identifier (e.g. 'D0', 'D1', 'D2', 'D3', 'D4')

    Returns:
        Driver status including location, route, assigned orders, and ETA
    """
    if driver_id not in env._drivers:
        return f"ERROR: Driver '{driver_id}' not found"

    d = env._drivers[driver_id]
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


def query_order(env: "LogiChainEnvironment", order_id: str) -> str:
    """
    Get detailed status of an order.

    Args:
        env: The environment instance
        order_id: Order identifier (e.g. 'O0', 'O1')

    Returns:
        Order details including pickup, dropoff, deadline, and current status
    """
    if order_id not in env._orders:
        return f"ERROR: Order '{order_id}' not found"

    o = env._orders[order_id]
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


def query_network(env: "LogiChainEnvironment") -> str:
    """
    Get network topology and current traffic conditions.

    Args:
        env: The environment instance

    Returns:
        Network map with nodes, connections, and traffic multipliers
    """
    if not env._network:
        return "ERROR: Network not initialized"

    lines = ["=== NETWORK MAP ==="]
    for node_id in env._network.get_all_nodes():
        node = env._network.get_node(node_id)
        neighbors = env._network.get_neighbors(node_id)
        neighbor_info = []
        for n in neighbors:
            road = env._network.get_road(node_id, n)
            traffic = env._network.get_traffic(node_id, n)
            neighbor_info.append(f"{n}({road.distance}*{traffic:.1f})")
        lines.append(f"{node_id} ({node.node_type}): {', '.join(neighbor_info)}")
    return "\n".join(lines)
