from __future__ import annotations

import time

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    AlignmentEstimate,
    Observation,
)


class FakeVisualServoController:
    """
    Fake Visual Servo。

    每一步向 aligned_target_q 靠近。

    真实版本以后会根据视觉误差
    计算 Cartesian / joint correction。
    """

    def __init__(
        self,
        aligned_target_q: np.ndarray,
        gain: float = 0.5,
    ):
        target = np.asarray(
            aligned_target_q,
            dtype=np.float64,
        )

        if target.ndim != 1:
            raise ValueError(
                "aligned_target_q must be 1-D"
            )

        if not 0 < gain <= 1:
            raise ValueError(
                "gain must be in (0, 1]"
            )

        self._target_q = target.copy()
        self._gain = gain

    def reset(self) -> None:
        pass

    def compute_action(
        self,
        observation: Observation,
        alignment: AlignmentEstimate,
    ) -> Action:

        if not alignment.visible:
            raise RuntimeError(
                "Cannot servo without visual target"
            )

        current_q = np.asarray(
            observation.robot.q,
            dtype=np.float64,
        )

        if current_q.shape != self._target_q.shape:
            raise ValueError(
                "robot q shape mismatch"
            )

        next_q = (
            current_q
            + self._gain
            * (
                self._target_q
                - current_q
            )
        )

        return Action(
            q=next_q,
            timestamp_s=time.monotonic(),
        )