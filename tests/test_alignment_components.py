import unittest

import numpy as np

from usb_insertion.core.datatypes import (
    FrameBundle,
    Observation,
    RobotState,
)
from usb_insertion.fakes.fake_alignment_estimator import (
    FakeAlignmentEstimator,
)
from usb_insertion.fakes.fake_visual_insertion import (
    FakeVisualInsertionController,
)
from usb_insertion.fakes.fake_visual_servo import (
    FakeVisualServoController,
)


class TestAlignmentComponents(
    unittest.TestCase
):
    def make_observation(
        self,
        q: np.ndarray,
    ) -> Observation:
        return Observation(
            robot=RobotState(
                q=q,
                timestamp_s=1.0,
            ),
            cameras=FrameBundle(
                frames={},
                timestamp_s=1.0,
            ),
        )

    def test_capture_region(self):
        estimator = (
            FakeAlignmentEstimator(
                capture_target_q=np.array([
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0.25,
                ]),
                aligned_target_q=np.array([
                    0.5,
                    0.2,
                    -0.1,
                    0.1,
                    0,
                    0.25,
                ]),
                capture_tolerance=0.05,
                aligned_tolerance=0.02,
            )
        )

        observation = (
            self.make_observation(
                np.array([
                    0.49,
                    0,
                    0,
                    0,
                    0,
                    0.25,
                ])
            )
        )

        result = estimator.estimate(
            observation
        )

        self.assertTrue(
            result.in_capture_region
        )

    def test_visual_servo_moves_to_target(
        self,
    ):
        target = np.ones(6)

        controller = (
            FakeVisualServoController(
                aligned_target_q=target,
                gain=0.5,
            )
        )

        estimator = (
            FakeAlignmentEstimator(
                capture_target_q=np.zeros(6),
                aligned_target_q=target,
                capture_tolerance=1,
                aligned_tolerance=0.01,
            )
        )

        observation = (
            self.make_observation(
                np.zeros(6)
            )
        )

        alignment = (
            estimator.estimate(
                observation
            )
        )

        action = (
            controller.compute_action(
                observation,
                alignment,
            )
        )

        np.testing.assert_allclose(
            action.q,
            np.full(6, 0.5),
        )

    def test_insertion_moves_single_axis(
        self,
    ):
        controller = (
            FakeVisualInsertionController(
                axis_index=0,
                target_position=1.0,
                step_size=0.1,
            )
        )

        observation = (
            self.make_observation(
                np.zeros(6)
            )
        )

        from usb_insertion.core.datatypes import (
            AlignmentEstimate,
        )

        alignment = AlignmentEstimate(
            visible=True,
            in_capture_region=True,
            aligned=True,
        )

        action = (
            controller.compute_action(
                observation,
                alignment,
            )
        )

        expected = np.zeros(6)
        expected[0] = 0.1

        np.testing.assert_allclose(
            action.q,
            expected,
        )


if __name__ == "__main__":
    unittest.main()