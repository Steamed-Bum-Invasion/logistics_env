# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
External grader for LogiChain environment tasks.
Provides deterministic scoring (0.0-1.0) based on task objectives.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from ..models import LogiChainState
except ImportError:
    from models import LogiChainState


@dataclass
class GraderResult:
    """Result from grading an episode."""

    score: float  # 0.0 to 1.0
    metrics: Dict[str, Any]
    done: bool = True


class TaskGrader:
    """
    External grader that evaluates episode performance.
    
    The grader runs AFTER the episode ends and provides a final score.
    This is separate from the internal reward signal the agent receives
    during the episode (shaping + terminal rewards).
    """

    def __init__(self, task_config: dict):
        self.tasks = task_config.get("tasks", {})
        self.default_task = task_config.get("default_task", "quick_delivery")
        self._current_task: Optional[str] = None

    def set_task(self, task_name: str) -> None:
        """Set the task to evaluate (must be in configured tasks)."""
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.tasks.keys())}")
        self._current_task = task_name

    def get_current_task(self) -> str:
        """Get current task name."""
        return self._current_task or self.default_task

    def grade(
        self,
        state: LogiChainState,
        orders: Dict[str, Dict[str, Any]],
        terminal_reward: float = 0.0,
    ) -> GraderResult:
        """
        Grade the episode based on the current task.
        
        Args:
            state: Final LogiChainState after episode ends
            orders: Final orders dict with status, deadline, etc.
            terminal_reward: Sum of terminal rewards from env (for reference)
        
        Returns:
            GraderResult with score (0.0-1.0) and metrics
        """
        task_name = self.get_current_task()
        task = self.tasks[task_name]
        grader_type = task.get("grader", {}).get("type", "count_based")
        weights = task.get("grader", {}).get("weights", {})

        if grader_type == "count_based":
            return self._grade_count_based(state, orders, weights)
        elif grader_type == "efficiency":
            return self._grade_efficiency(state, orders, weights)
        elif grader_type == "priority":
            return self._grade_priority(state, orders, weights)
        else:
            raise ValueError(f"Unknown grader type: {grader_type}")

    def _grade_count_based(
        self,
        state: LogiChainState,
        orders: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> GraderResult:
        """Count-based grading: simple delivered vs failed scoring."""
        delivered = state.orders_delivered
        failed = state.orders_failed

        # Calculate weighted score
        score = weights.get("delivered", 1.0) * delivered + weights.get("failed", -0.5) * failed

        # Normalize to 0-1 range (approximate, assuming typical range)
        # delivered: 0-20, failed: 0-10
        score = max(0.0, min(1.0, score / 10.0))

        return GraderResult(
            score=score,
            metrics={
                "delivered": delivered,
                "failed": failed,
                "on_time": state.on_time_deliveries,
            },
        )

    def _grade_efficiency(
        self,
        state: LogiChainState,
        orders: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> GraderResult:
        """Efficiency-based grading: on-time delivery focus."""
        delivered = state.orders_delivered
        failed = state.orders_failed
        on_time = state.on_time_deliveries

        # Calculate weighted score
        score = (
            weights.get("on_time", 0.6) * on_time
            + weights.get("delivered", 0.2) * delivered
            + weights.get("failed", -0.2) * failed
        )

        # Normalize (on_time is most important)
        # Typical: 0-15 on_time, 0-20 delivered, 0-10 failed
        score = max(0.0, min(1.0, score / 10.0))

        return GraderResult(
            score=score,
            metrics={
                "delivered": delivered,
                "failed": failed,
                "on_time": on_time,
                "efficiency": on_time / max(1, delivered + failed) if (delivered + failed) > 0 else 0.0,
            },
        )

    def _grade_priority(
        self,
        state: LogiChainState,
        orders: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> GraderResult:
        """Priority-based grading: focus on tight-deadline orders."""
        delivered = state.orders_delivered
        failed = state.orders_failed
        on_time = state.on_time_deliveries

        # Count high-priority deliveries (delivered before deadline)
        # and high-priority failures (failed with tight deadline)
        high_priority_delivered = 0
        high_priority_failed = 0

        for oid, order in orders.items():
            deadline = order.get("deadline", 100)
            if deadline <= 15:  # Tight deadline = high priority
                if order["status"] == "delivered":
                    high_priority_delivered += 1
                elif order["status"] == "failed":
                    high_priority_failed += 1

        # Calculate weighted score
        score = (
            weights.get("high_priority_delivered", 0.5) * high_priority_delivered
            + weights.get("on_time", 0.3) * on_time
            + weights.get("failed_high_priority", -0.5) * high_priority_failed
        )

        # Normalize (high_priority matters most)
        score = max(0.0, min(1.0, score / 8.0))

        return GraderResult(
            score=score,
            metrics={
                "delivered": delivered,
                "failed": failed,
                "on_time": on_time,
                "high_priority_delivered": high_priority_delivered,
                "high_priority_failed": high_priority_failed,
            },
        )
