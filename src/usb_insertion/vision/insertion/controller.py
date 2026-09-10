from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    AlignmentEstimate,
    Observation,
)
from usb_insertion.core.protocols import (
    VisualServoController,
)


class CalibratedVisualInsertionController:
    """
    视觉闭环插入控制器。

    total_joint_delta:
        从理想 Pre-insert 到正常插到底，
        通过后续标定获得：

            q_inserted - q_preinsert

    progress_step:
        每个控制周期推进整条插入路径的比例。

        例如：
            0.02
        大约需要 50 个周期走完整段。

    correction_servo:
        只使用 lateral/orientation error，
        不使用插入轴对应的 image error，
        避免 Visual Servo 把前进动作抵消。

    guard_error_limits:
        插入过程中允许的最大视觉误差。
        任何一项超出立即停止继续向前。
    """

    def __init__(
        self,
        *,
        total_joint_delta: np.ndarray,
        progress_step: float,
        correction_servo: VisualServoController,
        guard_error_limits: Mapping[
            str,
            float,
        ],
    ):
        delta = np.asarray(
            total_joint_delta,
            dtype=np.float64,
        )

        if delta.shape != (6,):
            raise ValueError(
                "total_joint_delta "
                "must have shape (6,)"
            )

        if not np.all(
            np.isfinite(delta)
        ):
            raise ValueError(
                "total_joint_delta "
                "contains NaN or Inf"
            )

        if not (
            0 < progress_step <= 1
        ):
            raise ValueError(
                "progress_step "
                "must be in (0, 1]"
            )

        for key, limit in (
            guard_error_limits.items()
        ):
            if float(limit) <= 0:
                raise ValueError(
                    "guard limit for "
                    f"{key} must be > 0"
                )

        self._total_joint_delta = (
            delta.copy()
        )

        self._progress_step = float(
            progress_step
        )

        self._correction_servo = (
            correction_servo
        )

        self._guard_error_limits = {
            str(key): float(value)
            for key, value
            in guard_error_limits.items()
        }

        self._progress = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    def reset(self) -> None:
        self._progress = 0.0

        self._correction_servo.reset()

    def is_safe(
        self,
        alignment: AlignmentEstimate,
    ) -> bool:

        if not alignment.visible:
            return False

        for (
            key,
            limit,
        ) in (
            self._guard_error_limits.items()
        ):
            if key not in (
                alignment.errors
            ):
                return False

            if abs(
                float(
                    alignment.errors[
                        key
                    ]
                )
            ) > limit:
                return False

        return True

    def compute_action(
        self,
        observation: Observation,
        alignment: AlignmentEstimate,
    ) -> Action:

        if not self.is_safe(
            alignment
        ):
            raise RuntimeError(
                "Insertion visual guard "
                "is not satisfied"
            )

        current_q = np.asarray(
            observation.robot.q,
            dtype=np.float64,
        )

        correction_action = (
            self._correction_servo
            .compute_action(
                observation,
                alignment,
            )
        )

        correction_q = np.asarray(
            correction_action.q,
            dtype=np.float64,
        )

        correction_delta = (
            correction_q
            - current_q
        )

        remaining = (
            1.0
            - self._progress
        )

        alpha_step = min(
            self._progress_step,
            remaining,
        )

        forward_delta = (
            self._total_joint_delta
            * alpha_step
        )

        next_q = (
            current_q
            + forward_delta
            + correction_delta
        )

        self._progress += (
            alpha_step
        )

        return Action(
            q=next_q,
            timestamp_s=time.monotonic(),
        )
