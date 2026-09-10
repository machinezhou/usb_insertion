import tempfile
import unittest
from pathlib import Path

import numpy as np

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
    FrameBundle,
    Observation,
    RobotState,
)
from usb_insertion.vision.alignment.config import (
    VisualServoConfig,
)
from usb_insertion.vision.alignment.servo import (
    JacobianVisualServoController,
)


class TestVisualServo(
    unittest.TestCase
):
    def test_identity_servo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "j.npy"
            )

            # 3 errors / 3 joints
            np.save(
                path,
                np.eye(3),
            )

            config = (
                VisualServoConfig(
                    error_keys=(
                        "e0",
                        "e1",
                        "e2",
                    ),
                    controlled_joint_indices=(
                        0,
                        1,
                        2,
                    ),
                    jacobian_path=path,
                    gain=0.5,
                    damping=0.0,
                    max_joint_delta=(
                        10,
                        10,
                        10,
                    ),
                )
            )

            controller = (
                JacobianVisualServoController(
                    config
                )
            )

            observation = Observation(
                robot=RobotState(
                    q=np.zeros(6),
                    timestamp_s=1,
                ),
                cameras=FrameBundle(
                    frames={},
                    timestamp_s=1,
                ),
            )

            alignment = (
                AlignmentEstimate(
                    visible=True,
                    in_capture_region=True,
                    aligned=False,
                    errors={
                        "e0": 2.0,
                        "e1": -4.0,
                        "e2": 1.0,
                    },
                )
            )

            action = (
                controller.compute_action(
                    observation,
                    alignment,
                )
            )

            expected = np.zeros(
                6
            )

            expected[0] = -1.0
            expected[1] = 2.0
            expected[2] = -0.5

            np.testing.assert_allclose(
                action.q,
                expected,
            )

    def test_delta_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "j.npy"
            )

            np.save(
                path,
                np.eye(1),
            )

            config = (
                VisualServoConfig(
                    error_keys=("e",),
                    controlled_joint_indices=(
                        0,
                    ),
                    jacobian_path=path,
                    gain=1.0,
                    damping=0.0,
                    max_joint_delta=(
                        0.2,
                    ),
                )
            )

            controller = (
                JacobianVisualServoController(
                    config
                )
            )

            observation = Observation(
                robot=RobotState(
                    q=np.zeros(6),
                    timestamp_s=1,
                ),
                cameras=FrameBundle(
                    frames={},
                    timestamp_s=1,
                ),
            )

            alignment = (
                AlignmentEstimate(
                    visible=True,
                    in_capture_region=True,
                    aligned=False,
                    errors={
                        "e": 100.0
                    },
                )
            )

            action = (
                controller.compute_action(
                    observation,
                    alignment,
                )
            )

            self.assertAlmostEqual(
                action.q[0],
                -0.2,
            )

    def test_select_subset_of_full_jacobian(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "j.npy"
            )

            full = np.array([
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 0.0],
            ])

            np.save(
                path,
                full,
            )

            config = (
                VisualServoConfig(
                    error_keys=(
                        "e1",
                        "e2",
                    ),
                    controlled_joint_indices=(
                        0,
                        1,
                    ),
                    jacobian_path=path,
                    gain=1.0,
                    damping=0.0,
                    max_joint_delta=(
                        10.0,
                        10.0,
                    ),
                )
            )

            controller = (
                JacobianVisualServoController(
                    config,
                    jacobian_error_keys=(
                        "e0",
                        "e1",
                        "e2",
                    ),
                    jacobian_controlled_joint_indices=(
                        0,
                        1,
                    ),
                )
            )

            np.testing.assert_allclose(
                controller.jacobian,
                full[[1, 2], :],
            )

if __name__ == "__main__":
    unittest.main()