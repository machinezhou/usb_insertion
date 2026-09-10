from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from usb_insertion.control.trajectory import (
    joint_trajectory,
)


# 对五次 smoothstep：
#
# s(u) = 10u^3 - 15u^4 + 6u^5
#
# 最大归一化速度：
# max(ds/du) = 15 / 8 = 1.875
#
# 最大归一化加速度：
# max(|d²s/du²|) = 10*sqrt(3)/3
#
QUINTIC_MAX_VELOCITY_FACTOR = 15.0 / 8.0

QUINTIC_MAX_ACCELERATION_FACTOR = (
    10.0 * math.sqrt(3.0) / 3.0
)


@dataclass(slots=True)
class TrajectoryLimits:
    """
    关节轨迹规划约束。

    所有 ndarray 均要求：

        shape = (dof,)

    max_velocity:
        每个关节允许的最大速度。

        当前 SO-101 使用 use_degrees=True 时，
        未来单位应为 degree / second。

    max_acceleration:
        每个关节允许的最大加速度。

        对应 degree / second^2。

    max_delta_per_step:
        每个控制周期允许的最大位置变化。

        后续应和 ActionGuard 使用同一组安全配置。
    """

    max_velocity: np.ndarray | None = None
    max_acceleration: np.ndarray | None = None
    max_delta_per_step: np.ndarray | None = None


@dataclass(slots=True)
class TrajectoryPlan:
    """
    一次轨迹规划结果。
    """

    duration_s: float
    trajectory: np.ndarray

    @property
    def command_steps(self) -> int:
        """
        第一个点是当前状态，
        因此真正需要发送的 command 数量少 1。
        """
        return max(
            0,
            self.trajectory.shape[0] - 1,
        )


def _validate_joint_vector(
    name: str,
    value: np.ndarray,
    expected_shape: tuple[int, ...],
    *,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array"
        )

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch: "
            f"{array.shape} != {expected_shape}"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{name} contains NaN or Inf"
        )

    if positive and np.any(
        array <= 0
    ):
        raise ValueError(
            f"{name} must contain only "
            "positive values"
        )

    return array


def _round_duration_to_control_period(
    duration_s: float,
    fps: float,
) -> float:
    """
    把 duration 向上取整到整数个控制周期。

    例如：

        fps = 30 Hz

    一个周期：

        1 / 30 second

    这样最后的轨迹点能和控制周期严格对齐。
    """

    periods = math.ceil(
        duration_s * fps - 1e-12
    )

    periods = max(
        1,
        periods,
    )

    return periods / fps


def minimum_duration(
    q_start: np.ndarray,
    q_target: np.ndarray,
    fps: float,
    limits: TrajectoryLimits,
    requested_duration_s: float | None = None,
) -> float:
    """
    根据五次 smoothstep 的速度、加速度和单步变化约束，
    计算轨迹至少需要多长时间。

    requested_duration_s 并不是强制最终时间。

    如果用户要求 2 秒，但安全约束要求至少 3 秒：

        最终使用 3 秒。

    如果约束只要求 1 秒，而用户希望 3 秒：

        最终使用 3 秒。
    """

    if fps <= 0:
        raise ValueError(
            "fps must be greater than 0"
        )

    q_start = np.asarray(
        q_start,
        dtype=np.float64,
    )

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    )

    if q_start.ndim != 1:
        raise ValueError(
            "q_start must be a 1-D array"
        )

    if q_target.ndim != 1:
        raise ValueError(
            "q_target must be a 1-D array"
        )

    if q_start.shape != q_target.shape:
        raise ValueError(
            "q_start and q_target must "
            "have the same shape"
        )

    if not np.all(np.isfinite(q_start)):
        raise ValueError(
            "q_start contains NaN or Inf"
        )

    if not np.all(np.isfinite(q_target)):
        raise ValueError(
            "q_target contains NaN or Inf"
        )

    if requested_duration_s is not None:
        if (
            not math.isfinite(
                requested_duration_s
            )
            or requested_duration_s <= 0
        ):
            raise ValueError(
                "requested_duration_s "
                "must be > 0"
            )

    delta = np.abs(
        q_target - q_start
    )

    required_duration = 0.0

    if requested_duration_s is not None:
        required_duration = max(
            required_duration,
            requested_duration_s,
        )

    # ---------------------------------------------------------
    # Velocity constraint
    #
    # v_max =
    #   QUINTIC_MAX_VELOCITY_FACTOR * delta / T
    #
    # 所以：
    #
    # T >= factor * delta / velocity_limit
    # ---------------------------------------------------------

    if limits.max_velocity is not None:
        max_velocity = _validate_joint_vector(
            "max_velocity",
            limits.max_velocity,
            q_start.shape,
            positive=True,
        )

        velocity_duration = (
            QUINTIC_MAX_VELOCITY_FACTOR
            * delta
            / max_velocity
        )

        required_duration = max(
            required_duration,
            float(
                np.max(
                    velocity_duration
                )
            ),
        )

    # ---------------------------------------------------------
    # Acceleration constraint
    #
    # a_max =
    #   factor * delta / T^2
    #
    # 所以：
    #
    # T >= sqrt(factor * delta / acceleration_limit)
    # ---------------------------------------------------------

    if limits.max_acceleration is not None:
        max_acceleration = (
            _validate_joint_vector(
                "max_acceleration",
                limits.max_acceleration,
                q_start.shape,
                positive=True,
            )
        )

        acceleration_duration = np.sqrt(
            QUINTIC_MAX_ACCELERATION_FACTOR
            * delta
            / max_acceleration
        )

        required_duration = max(
            required_duration,
            float(
                np.max(
                    acceleration_duration
                )
            ),
        )

    # ---------------------------------------------------------
    # Per-step delta constraint
    #
    # 一个控制周期允许的最大位置变化：
    #
    # max_delta_per_step
    #
    # 它可以近似看作：
    #
    # equivalent_velocity =
    #     max_delta_per_step * fps
    #
    # 先用连续速度公式得到一个保守初值。
    # 最终 plan_joint_trajectory() 还会进行实际离散检查。
    # ---------------------------------------------------------

    if limits.max_delta_per_step is not None:
        max_delta = _validate_joint_vector(
            "max_delta_per_step",
            limits.max_delta_per_step,
            q_start.shape,
            positive=True,
        )

        equivalent_velocity = (
            max_delta * fps
        )

        delta_duration = (
            QUINTIC_MAX_VELOCITY_FACTOR
            * delta
            / equivalent_velocity
        )

        required_duration = max(
            required_duration,
            float(
                np.max(
                    delta_duration
                )
            ),
        )

    # 完全没有移动、也没有 requested duration 时，
    # 至少返回一个控制周期。
    if required_duration <= 0:
        required_duration = (
            1.0 / fps
        )

    return (
        _round_duration_to_control_period(
            required_duration,
            fps,
        )
    )


def plan_joint_trajectory(
    q_start: np.ndarray,
    q_target: np.ndarray,
    fps: float,
    limits: TrajectoryLimits,
    requested_duration_s: float | None = None,
) -> TrajectoryPlan:
    """
    规划一条满足约束的五次平滑关节轨迹。

    除了连续速度/加速度计算外，
    还会对最终离散轨迹逐步检查
    max_delta_per_step。
    """

    q_start = np.asarray(
        q_start,
        dtype=np.float64,
    )

    q_target = np.asarray(
        q_target,
        dtype=np.float64,
    )

    duration_s = minimum_duration(
        q_start=q_start,
        q_target=q_target,
        fps=fps,
        limits=limits,
        requested_duration_s=(
            requested_duration_s
        ),
    )

    control_period_s = (
        1.0 / fps
    )

    max_delta = None

    if limits.max_delta_per_step is not None:
        max_delta = _validate_joint_vector(
            "max_delta_per_step",
            limits.max_delta_per_step,
            q_start.shape,
            positive=True,
        )

    # 通常第一次就会满足。
    #
    # 这里仍然做实际离散验证，
    # 避免因为采样/rounding 导致
    # 某一个 waypoint 超出 per-step limit。
    for _ in range(10000):
        trajectory = joint_trajectory(
            q_start=q_start,
            q_target=q_target,
            duration_s=duration_s,
            fps=fps,
        )

        if max_delta is None:
            return TrajectoryPlan(
                duration_s=duration_s,
                trajectory=trajectory,
            )

        step_delta = np.abs(
            np.diff(
                trajectory,
                axis=0,
            )
        )

        if np.all(
            step_delta
            <= max_delta[None, :] + 1e-12
        ):
            return TrajectoryPlan(
                duration_s=duration_s,
                trajectory=trajectory,
            )

        # 如果离散检查仍不满足，
        # 再增加一个控制周期。
        duration_s += (
            control_period_s
        )

    raise RuntimeError(
        "Unable to find a trajectory "
        "satisfying max_delta_per_step"
    )