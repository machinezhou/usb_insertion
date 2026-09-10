from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    AlignmentEstimate,
    Observation,
)
from usb_insertion.vision.alignment.config import (
    VisualServoConfig,
)
from usb_insertion.vision.alignment.jacobian import (
    load_image_jacobian,
)


class JacobianVisualServoController:
    """
    基于局部 Image Jacobian 的视觉伺服。

    允许从完整 Jacobian 中选择：

        部分 error rows
        部分 joint columns

    因此：

    Alignment：
        可以使用完整 error vector。

    Insertion：
        可以只使用 lateral/orientation error，
        不把插入轴前进量抵消掉。
    """

    def __init__(
        self,
        config: VisualServoConfig,
        *,
        jacobian_error_keys: (
            Sequence[str] | None
        ) = None,
        jacobian_controlled_joint_indices: (
            Sequence[int] | None
        ) = None,
    ):
        self.config = config

        source_error_keys = tuple(
            config.error_keys
            if jacobian_error_keys is None
            else (
                str(value)
                for value
                in jacobian_error_keys
            )
        )

        source_joint_indices = tuple(
            config
            .controlled_joint_indices
            if (
                jacobian_controlled_joint_indices
                is None
            )
            else (
                int(value)
                for value
                in (
                    jacobian_controlled_joint_indices
                )
            )
        )

        model = load_image_jacobian(
            config.jacobian_path,
            error_keys=(
                source_error_keys
            ),
            controlled_joint_indices=(
                source_joint_indices
            ),
        )

        missing_errors = [
            key
            for key
            in config.error_keys
            if key not in (
                source_error_keys
            )
        ]

        if missing_errors:
            raise ValueError(
                "requested servo error keys "
                "are not in Jacobian: "
                f"{missing_errors}"
            )

        missing_joints = [
            index
            for index
            in config
            .controlled_joint_indices
            if index not in (
                source_joint_indices
            )
        ]

        if missing_joints:
            raise ValueError(
                "requested controlled joints "
                "are not in Jacobian: "
                f"{missing_joints}"
            )

        row_indices = [
            source_error_keys.index(
                key
            )
            for key
            in config.error_keys
        ]

        column_indices = [
            source_joint_indices.index(
                index
            )
            for index
            in config
            .controlled_joint_indices
        ]

        self._jacobian = (
            model.matrix[
                np.ix_(
                    row_indices,
                    column_indices,
                )
            ]
            .copy()
        )

        expected_shape = (
            len(
                config.error_keys
            ),
            len(
                config
                .controlled_joint_indices
            ),
        )

        if (
            self._jacobian.shape
            != expected_shape
        ):
            raise ValueError(
                "Jacobian shape mismatch: "
                f"{self._jacobian.shape} "
                f"!= {expected_shape}"
            )

        if config.gain <= 0:
            raise ValueError(
                "servo gain must be > 0"
            )

        if config.damping < 0:
            raise ValueError(
                "servo damping must be >= 0"
            )

        self._max_delta = np.asarray(
            config.max_joint_delta,
            dtype=np.float64,
        )

        expected_delta_shape = (
            len(
                config
                .controlled_joint_indices
            ),
        )

        if (
            self._max_delta.shape
            != expected_delta_shape
        ):
            raise ValueError(
                "max_joint_delta "
                "shape mismatch"
            )

        if not np.all(
            np.isfinite(
                self._max_delta
            )
        ):
            raise ValueError(
                "max_joint_delta contains "
                "NaN or Inf"
            )

        if np.any(
            self._max_delta <= 0
        ):
            raise ValueError(
                "max_joint_delta values "
                "must be > 0"
            )

    @property
    def jacobian(
        self,
    ) -> np.ndarray:
        return self._jacobian.copy()

    def reset(self) -> None:
        pass

    def _error_vector(
        self,
        alignment: AlignmentEstimate,
    ) -> np.ndarray:

        values = []

        for key in (
            self.config.error_keys
        ):
            if key not in (
                alignment.errors
            ):
                raise KeyError(
                    "AlignmentEstimate "
                    f"missing error: {key}"
                )

            value = float(
                alignment.errors[
                    key
                ]
            )

            if not np.isfinite(
                value
            ):
                raise ValueError(
                    f"Alignment error "
                    f"'{key}' is NaN or Inf"
                )

            values.append(
                value
            )

        return np.asarray(
            values,
            dtype=np.float64,
        )

    def _damped_pseudoinverse(
        self,
    ) -> np.ndarray:

        j = self._jacobian

        damping = float(
            self.config.damping
        )

        if damping == 0:
            return np.linalg.pinv(
                j
            )

        damping_sq = (
            damping
            * damping
        )

        if (
            j.shape[0]
            <= j.shape[1]
        ):
            identity = np.eye(
                j.shape[0],
                dtype=np.float64,
            )

            return (
                j.T
                @ np.linalg.solve(
                    (
                        j @ j.T
                        + damping_sq
                        * identity
                    ),
                    identity,
                )
            )

        identity = np.eye(
            j.shape[1],
            dtype=np.float64,
        )

        return np.linalg.solve(
            (
                j.T @ j
                + damping_sq
                * identity
            ),
            j.T,
        )

    def compute_action(
        self,
        observation: Observation,
        alignment: AlignmentEstimate,
    ) -> Action:

        if not alignment.visible:
            raise RuntimeError(
                "Cannot visual-servo "
                "without visible USB"
            )

        current_q = np.asarray(
            observation.robot.q,
            dtype=np.float64,
        )

        if current_q.ndim != 1:
            raise ValueError(
                "robot q must be 1-D"
            )

        if not np.all(
            np.isfinite(
                current_q
            )
        ):
            raise ValueError(
                "robot q contains NaN or Inf"
            )

        error = self._error_vector(
            alignment
        )

        pseudo_inverse = (
            self
            ._damped_pseudoinverse()
        )

        dq = (
            -float(
                self.config.gain
            )
            * (
                pseudo_inverse
                @ error
            )
        )

        if not np.all(
            np.isfinite(dq)
        ):
            raise ValueError(
                "Visual Servo produced "
                "NaN or Inf"
            )

        dq = np.clip(
            dq,
            -self._max_delta,
            self._max_delta,
        )

        next_q = current_q.copy()

        for (
            local_index,
            joint_index,
        ) in enumerate(
            self.config
            .controlled_joint_indices
        ):
            if (
                joint_index < 0
                or joint_index
                >= next_q.size
            ):
                raise ValueError(
                    "controlled joint index "
                    "outside robot q: "
                    f"{joint_index}"
                )

            next_q[
                joint_index
            ] += dq[
                local_index
            ]

        return Action(
            q=next_q,
            timestamp_s=time.monotonic(),
        )
