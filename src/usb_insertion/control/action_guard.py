from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usb_insertion.core.datatypes import Action


@dataclass(slots=True)
class ActionGuardConfig:
    """
    ActionGuard 配置。

    gripper_index:
        gripper 在关节数组中的索引。
        SO-101 最终确认后再固定具体值。

    fixed_gripper_position:
        USB 固定后，gripper 应保持的位置。

    max_delta_per_step:
        每个控制周期允许的最大关节位置变化。
        shape = (dof,)

    joint_min / joint_max:
        每个关节允许的安全范围。
        shape = (dof,)
    """

    gripper_index: int | None = None
    fixed_gripper_position: float | None = None

    max_delta_per_step: np.ndarray | None = None

    joint_min: np.ndarray | None = None
    joint_max: np.ndarray | None = None


class ActionGuard:
    """
    所有机器人动作在发送给真实机器人之前，
    都必须经过这一层。
    """

    def __init__(self, config: ActionGuardConfig):
        self.config = config

    def apply(
        self,
        raw_action: Action,
        current_q: np.ndarray,
    ) -> Action:
        """
        对目标 action 做安全处理。

        处理顺序：

        1. 检查 shape
        2. 检查 NaN / Inf
        3. 覆盖 gripper target
        4. 限制单周期 delta
        5. 限制 joint range
        """

        target_q = np.asarray(
            raw_action.q,
            dtype=np.float64,
        ).copy()

        current_q = np.asarray(
            current_q,
            dtype=np.float64,
        )

        if target_q.ndim != 1:
            raise ValueError("raw_action.q must be a 1-D array")

        if current_q.ndim != 1:
            raise ValueError("current_q must be a 1-D array")

        if target_q.shape != current_q.shape:
            raise ValueError(
                "action and current state must have the same shape: "
                f"{target_q.shape} != {current_q.shape}"
            )

        if not np.all(np.isfinite(target_q)):
            raise ValueError(
                "raw_action.q contains NaN or Inf"
            )

        if not np.all(np.isfinite(current_q)):
            raise ValueError(
                "current_q contains NaN or Inf"
            )

        # ---------------------------------------------------------
        # 固定 gripper
        # ---------------------------------------------------------

        if (
            self.config.gripper_index is not None
            and self.config.fixed_gripper_position is not None
        ):
            gripper_index = self.config.gripper_index

            if not 0 <= gripper_index < target_q.size:
                raise ValueError(
                    f"invalid gripper_index: {gripper_index}"
                )

            target_q[gripper_index] = (
                self.config.fixed_gripper_position
            )

        # ---------------------------------------------------------
        # 单周期动作限幅
        # ---------------------------------------------------------

        if self.config.max_delta_per_step is not None:
            max_delta = np.asarray(
                self.config.max_delta_per_step,
                dtype=np.float64,
            )

            if max_delta.shape != target_q.shape:
                raise ValueError(
                    "max_delta_per_step shape mismatch"
                )

            if np.any(max_delta < 0):
                raise ValueError(
                    "max_delta_per_step must be non-negative"
                )

            delta = target_q - current_q

            delta = np.clip(
                delta,
                -max_delta,
                max_delta,
            )

            target_q = current_q + delta

        # ---------------------------------------------------------
        # Joint limits
        # ---------------------------------------------------------

        if self.config.joint_min is not None:
            joint_min = np.asarray(
                self.config.joint_min,
                dtype=np.float64,
            )

            if joint_min.shape != target_q.shape:
                raise ValueError(
                    "joint_min shape mismatch"
                )

            target_q = np.maximum(
                target_q,
                joint_min,
            )

        if self.config.joint_max is not None:
            joint_max = np.asarray(
                self.config.joint_max,
                dtype=np.float64,
            )

            if joint_max.shape != target_q.shape:
                raise ValueError(
                    "joint_max shape mismatch"
                )

            target_q = np.minimum(
                target_q,
                joint_max,
            )

        return Action(
            q=target_q,
            timestamp_s=raw_action.timestamp_s,
        )