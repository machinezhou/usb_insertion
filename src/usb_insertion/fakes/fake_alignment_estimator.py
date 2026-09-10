from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
    Observation,
)


class FakeAlignmentEstimator:
    """
    Fake 视觉估计器。

    不看真实图像。

    通过 FakeRobot 的 q 判断：
        是否进入 capture region
        是否完成 alignment

    仅用于验证业务架构。
    """

    def __init__(
        self,
        capture_target_q: np.ndarray,
        aligned_target_q: np.ndarray,
        capture_tolerance: float = 0.05,
        aligned_tolerance: float = 0.02,
        alignment_indices: Sequence[int] = (
            1,
            2,
            3,
            4,
        ),
    ):
        capture_target_q = np.asarray(
            capture_target_q,
            dtype=np.float64,
        )

        aligned_target_q = np.asarray(
            aligned_target_q,
            dtype=np.float64,
        )

        if (
            capture_target_q.ndim != 1
            or aligned_target_q.ndim != 1
        ):
            raise ValueError(
                "targets must be 1-D arrays"
            )

        if (
            capture_target_q.shape
            != aligned_target_q.shape
        ):
            raise ValueError(
                "target shapes must match"
            )

        if capture_tolerance <= 0:
            raise ValueError(
                "capture_tolerance must be > 0"
            )

        if aligned_tolerance <= 0:
            raise ValueError(
                "aligned_tolerance must be > 0"
            )

        indices = tuple(
            int(index)
            for index in alignment_indices
        )

        if not indices:
            raise ValueError(
                "alignment_indices must not be empty"
            )

        for index in indices:
            if not 0 <= index < capture_target_q.size:
                raise ValueError(
                    f"invalid alignment index: {index}"
                )

        self._capture_target_q = (
            capture_target_q.copy()
        )

        self._aligned_target_q = (
            aligned_target_q.copy()
        )

        self._capture_tolerance = (
            capture_tolerance
        )

        self._aligned_tolerance = (
            aligned_tolerance
        )

        self._alignment_indices = indices

    def reset(self) -> None:
        pass

    def estimate(
        self,
        observation: Observation,
    ) -> AlignmentEstimate:

        q = np.asarray(
            observation.robot.q,
            dtype=np.float64,
        )

        if q.shape != self._capture_target_q.shape:
            raise ValueError(
                "robot q shape mismatch"
            )

        capture_error = float(
            np.linalg.norm(
                q - self._capture_target_q
            )
        )

        indices = np.asarray(
            self._alignment_indices,
            dtype=int,
        )

        alignment_error = float(
            np.linalg.norm(
                q[indices]
                - self._aligned_target_q[
                    indices
                ]
            )
        )

        in_capture_region = (
            capture_error
            <= self._capture_tolerance
        )

        aligned = (
            alignment_error
            <= self._aligned_tolerance
        )

        return AlignmentEstimate(
            visible=True,
            in_capture_region=(
                in_capture_region
            ),
            aligned=aligned,
            errors={
                "fake_capture_error": (
                    capture_error
                ),
                "fake_alignment_error": (
                    alignment_error
                ),
            },
            confidence=1.0,
            plug_face="marked",
            reason=(
                "fake_aligned"
                if aligned
                else "fake_visible"
            ),
        )