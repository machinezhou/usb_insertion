from __future__ import annotations

import time

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    AlignmentEstimate,
    Observation,
)


class FakeVisualInsertionController:
    def __init__(
        self,
        axis_index: int,
        target_position: float,
        step_size: float,
    ):
        if axis_index < 0:
            raise ValueError(
                "axis_index must be >= 0"
            )

        if step_size <= 0:
            raise ValueError(
                "step_size must be > 0"
            )

        self._axis_index = axis_index
        self._target_position = float(
            target_position
        )
        self._step_size = float(
            step_size
        )

    def reset(self) -> None:
        pass

    def is_safe(
        self,
        alignment: AlignmentEstimate,
    ) -> bool:
        return (
            alignment.visible
            and alignment.aligned
        )

    def compute_action(
        self,
        observation: Observation,
        alignment: AlignmentEstimate,
    ) -> Action:

        if not self.is_safe(
            alignment
        ):
            raise RuntimeError(
                "Fake insertion unsafe"
            )

        q = np.asarray(
            observation.robot.q,
            dtype=np.float64,
        ).copy()

        if self._axis_index >= q.size:
            raise ValueError(
                "axis_index outside q"
            )

        current = q[
            self._axis_index
        ]

        error = (
            self._target_position
            - current
        )

        if abs(error) <= (
            self._step_size
        ):
            q[
                self._axis_index
            ] = (
                self._target_position
            )
        else:
            q[
                self._axis_index
            ] = (
                current
                + np.sign(error)
                * self._step_size
            )

        return Action(
            q=q,
            timestamp_s=time.monotonic(),
        )
