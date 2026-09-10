import unittest

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    AlignmentEstimate,
    FrameBundle,
    Observation,
    RobotState,
)
from usb_insertion.vision.insertion.controller import (
    CalibratedVisualInsertionController,
)


class ZeroCorrectionServo:
    def reset(self):
        pass

    def compute_action(
        self,
        observation,
        alignment,
    ):
        return Action(
            q=observation
            .robot
            .q
            .copy()
        )


class TestVisualInsertionController(
    unittest.TestCase
):
    def make_observation(self):
        return Observation(
            robot=RobotState(
                q=np.zeros(6),
                timestamp_s=1.0,
            ),
            cameras=FrameBundle(
                frames={},
                timestamp_s=1.0,
            ),
        )

    def test_forward_progress(self):
        controller = (
            CalibratedVisualInsertionController(
                total_joint_delta=np.asarray([
                    1,
                    2,
                    3,
                    4,
                    5,
                    0,
                ]),
                progress_step=0.1,
                correction_servo=(
                    ZeroCorrectionServo()
                ),
                guard_error_limits={
                    "lateral": 5.0,
                },
            )
        )

        alignment = (
            AlignmentEstimate(
                visible=True,
                in_capture_region=True,
                aligned=True,
                errors={
                    "lateral": 0.0
                },
            )
        )

        action = (
            controller.compute_action(
                self.make_observation(),
                alignment,
            )
        )

        np.testing.assert_allclose(
            action.q,
            np.asarray([
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.0,
            ]),
        )

        self.assertAlmostEqual(
            controller.progress,
            0.1,
        )

    def test_guard(self):
        controller = (
            CalibratedVisualInsertionController(
                total_joint_delta=np.ones(6),
                progress_step=0.1,
                correction_servo=(
                    ZeroCorrectionServo()
                ),
                guard_error_limits={
                    "lateral": 5,
                },
            )
        )

        alignment = (
            AlignmentEstimate(
                visible=True,
                in_capture_region=True,
                aligned=False,
                errors={
                    "lateral": 6,
                },
            )
        )

        self.assertFalse(
            controller.is_safe(
                alignment
            )
        )


if __name__ == "__main__":
    unittest.main()
