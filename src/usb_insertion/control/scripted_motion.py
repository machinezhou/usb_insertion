from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from usb_insertion.control.action_guard import (
    ActionGuard,
)
from usb_insertion.control.trajectory_planner import (
    TrajectoryLimits,
    plan_joint_trajectory,
)
from usb_insertion.core.datatypes import Action
from usb_insertion.core.protocols import (
    RobotBackend,
)


@dataclass(slots=True)
class MotionResult:
    """
    一次确定性关节运动执行结果。
    """

    start_q: np.ndarray
    target_q: np.ndarray
    final_command_q: np.ndarray

    planned_duration_s: float
    command_steps: int


class ScriptedMotionController:
    """
    确定性关节运动控制器。

    用于：
    - Home
    - Approach
    - Pre-insert
    - Retry retract

    不用于 ACT 精插入阶段。
    """

    def __init__(
        self,
        robot: RobotBackend,
        action_guard: ActionGuard,
        control_fps: float = 30.0,
        trajectory_limits: TrajectoryLimits | None = None,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if control_fps <= 0:
            raise ValueError(
                "control_fps must be greater than 0"
            )

        self._robot = robot
        self._action_guard = action_guard
        self._control_fps = control_fps

        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn

        self._trajectory_limits = (
            self._make_effective_limits(
                trajectory_limits
            )
        )

    @property
    def control_fps(self) -> float:
        return self._control_fps

    @property
    def trajectory_limits(
        self,
    ) -> TrajectoryLimits:
        return self._trajectory_limits

    def _make_effective_limits(
        self,
        limits: TrajectoryLimits | None,
    ) -> TrajectoryLimits:
        """
        Planner 的 max_delta 不能比 ActionGuard 更宽松。

        如果两边都配置：
            使用更严格的那个。

        如果 Planner 未配置 max_delta，
        但 ActionGuard 配置了：
            自动继承 ActionGuard。
        """

        if limits is None:
            limits = TrajectoryLimits()

        planner_delta = (
            None
            if limits.max_delta_per_step is None
            else np.asarray(
                limits.max_delta_per_step,
                dtype=np.float64,
            ).copy()
        )

        guard_delta = (
            self._action_guard
            .config
            .max_delta_per_step
        )

        if guard_delta is not None:
            guard_delta = np.asarray(
                guard_delta,
                dtype=np.float64,
            ).copy()

            if planner_delta is None:
                planner_delta = guard_delta

            else:
                if (
                    planner_delta.shape
                    != guard_delta.shape
                ):
                    raise ValueError(
                        "TrajectoryPlanner and "
                        "ActionGuard max_delta "
                        "shape mismatch"
                    )

                planner_delta = np.minimum(
                    planner_delta,
                    guard_delta,
                )

        return TrajectoryLimits(
            max_velocity=(
                None
                if limits.max_velocity is None
                else np.asarray(
                    limits.max_velocity,
                    dtype=np.float64,
                ).copy()
            ),
            max_acceleration=(
                None
                if limits.max_acceleration is None
                else np.asarray(
                    limits.max_acceleration,
                    dtype=np.float64,
                ).copy()
            ),
            max_delta_per_step=planner_delta,
        )

    def move_to(
        self,
        target_q: np.ndarray,
        duration_s: float | None = None,
        realtime: bool = True,
    ) -> MotionResult:
        """
        从当前 q 平滑移动到 target_q。

        duration_s:
            用户希望的最短时间。

            Planner 可以因为安全约束，
            自动把最终时间延长。
        """

        start_state = (
            self._robot.get_state()
        )

        start_q = np.asarray(
            start_state.q,
            dtype=np.float64,
        ).copy()

        target_q = np.asarray(
            target_q,
            dtype=np.float64,
        ).copy()

        plan = plan_joint_trajectory(
            q_start=start_q,
            q_target=target_q,
            fps=self._control_fps,
            limits=self._trajectory_limits,
            requested_duration_s=duration_s,
        )

        command_trajectory = (
            plan.trajectory[1:]
        )

        reference_q = (
            start_q.copy()
        )

        start_time = (
            self._clock_fn()
        )

        control_period_s = (
            1.0 / self._control_fps
        )

        for step_index, waypoint_q in enumerate(
            command_trajectory,
            start=1,
        ):
            if realtime:
                deadline = (
                    start_time
                    + step_index
                    * control_period_s
                )

                remaining = (
                    deadline
                    - self._clock_fn()
                )

                if remaining > 0:
                    self._sleep_fn(
                        remaining
                    )

            raw_action = Action(
                q=waypoint_q,
                timestamp_s=self._clock_fn(),
            )

            safe_action = (
                self._action_guard.apply(
                    raw_action=raw_action,
                    current_q=reference_q,
                )
            )

            self._robot.send_action(
                safe_action
            )

            reference_q = (
                safe_action.q.copy()
            )

        return MotionResult(
            start_q=start_q,
            target_q=target_q,
            final_command_q=reference_q,
            planned_duration_s=(
                plan.duration_s
            ),
            command_steps=len(
                command_trajectory
            ),
        )