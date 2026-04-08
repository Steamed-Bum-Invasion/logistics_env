# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward computation for LogiChain environment.

Rewards are computed based on:
1. Terminal rewards - when orders resolve (delivered/failed)
2. Shaping rewards - based on actions taken
3. No-progress penalty - encourages agent to act
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logistics_env.models import LogiChainAction
    from logistics_env.server.logistics_environment import LogiChainEnvironment


def compute_step_reward(
    env: "LogiChainEnvironment",
    action: "LogiChainAction",
    tool_result: str,
    is_error: bool,
) -> float:
    """
    Compute total reward for the current step.

    Combines terminal rewards, shaping rewards, and no-progress penalty.

    Args:
        env: The environment instance
        action: The action taken (for shaping reward)
        tool_result: Result text from tool execution
        is_error: Whether the tool call resulted in an error

    Returns:
        Total reward for this step
    """
    terminal_reward = compute_terminal_reward(env)
    shaping_reward = compute_shaping_reward(env, action, tool_result, is_error)
    no_progress_penalty = check_no_progress_penalty(env)

    return terminal_reward + shaping_reward + no_progress_penalty


def compute_terminal_reward(env: "LogiChainEnvironment") -> float:
    """
    Compute terminal reward for orders that resolved this step.

    Terminal rewards are applied when:
    - Order is delivered (on-time or late)
    - Order fails (deadline exceeded)

    Returns:
        Sum of terminal rewards for this step
    """
    if not env._reward_config.get("terminal", {}).get("enabled", True):
        return 0.0

    cfg = env._reward_config["terminal"]
    deadline_scaling = cfg.get("deadline_scaling", True)
    reward = 0.0

    for o in env._orders.values():
        if o["status"] == "delivered" and o.get("delivered") == env._state.time_step:
            if deadline_scaling:
                deadline = o.get("deadline", 30)
                max_deadline = 30
                deadline_factor = 1.0 + (max_deadline - deadline) / max_deadline
            else:
                deadline_factor = 1.0

            if o["delivered"] <= o["deadline"]:
                reward += cfg.get("delivery_on_time", 1.0) * deadline_factor
            else:
                reward += cfg.get("delivery_late", 0.5) * deadline_factor

        elif o["status"] == "failed" and o.get("failed_at") == env._state.time_step:
            reward += cfg.get("order_failed", -0.5)

    return reward


def compute_shaping_reward(
    env: "LogiChainEnvironment",
    action: "LogiChainAction",
    tool_result: str,
    is_error: bool,
) -> float:
    """
    Compute shaping reward based on the action taken.

    Shaping rewards encourage:
    - Querying for information (small positive)
    - Taking action (assign, reroute)
    - Penalizing stalling (delay, escalate)
    - Penalizing errors
    - Penalizing driver reassignment (abandoning orders)

    Returns:
        Shaping reward for this action
    """
    cfg = env._reward_config.get("shaping", {})

    if action is None:
        return 0.0

    if is_error or (tool_result and tool_result.startswith("ERROR")):
        return cfg.get("error", -0.05)

    tool_name = action.tool_name

    if tool_name in ("query_driver", "query_order", "query_network"):
        return cfg.get("query", 0.01)

    if tool_name == "assign_order":
        if tool_result and "OK:" in tool_result:
            base_reward = cfg.get("assign", 0.02)
            if "Reassigned" in tool_result:
                reassign_penalty = cfg.get("reassign", -0.1)
                return base_reward + reassign_penalty
            return base_reward
        return 0.0

    if tool_name == "reroute_driver":
        if tool_result and "OK:" in tool_result:
            return cfg.get("reroute", 0.02)
        return 0.0

    if tool_name == "delay_order":
        return cfg.get("delay", -0.02)

    if tool_name == "escalate_order":
        return cfg.get("escalate", -0.05)

    return 0.0


def check_no_progress_penalty(env: "LogiChainEnvironment") -> float:
    """
    Check for no-progress and apply penalty if threshold exceeded.

    Progress is defined as:
    - New deliveries completed
    - New orders assigned or in transit

    If no progress for too many steps, apply penalty to encourage action.

    Returns:
        No-progress penalty (0.0 or negative)
    """
    cfg = env._reward_config.get("no_progress", {})
    threshold = cfg.get("threshold", 10)
    penalty = cfg.get("penalty", -0.05)

    current_deliveries = sum(
        1 for o in env._orders.values() if o["status"] == "delivered"
    )
    current_active = sum(
        1 for o in env._orders.values()
        if o["status"] in ("assigned", "in_transit")
    )

    has_progress = (
        current_deliveries > env._total_deliveries_at_start
        or current_active > env._total_assignments_at_start
    )

    if has_progress:
        env._last_progress_step = env._state.time_step
        env._total_deliveries_at_start = current_deliveries
        env._total_assignments_at_start = current_active
        env._steps_without_progress = 0
        return 0.0

    env._steps_without_progress += 1

    if env._steps_without_progress > threshold:
        return penalty

    return 0.0
