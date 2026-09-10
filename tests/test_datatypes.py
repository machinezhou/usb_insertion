import unittest

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    DetectionResult,
    FrameBundle,
    Observation,
    RobotState,
)


class TestDatatypes(unittest.TestCase):
    def test_robot_state(self):
        q = np.zeros(6)

        state = RobotState(
            q=q,
            timestamp_s=1.0,
        )

        self.assertEqual(state.q.shape, (6,))
        self.assertEqual(state.timestamp_s, 1.0)

    def test_observation(self):
        state = RobotState(
            q=np.zeros(6),
            timestamp_s=1.0,
        )

        frames = FrameBundle(
            frames={
                "top": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "side": np.zeros((480, 640, 3), dtype=np.uint8),
            },
            timestamp_s=1.0,
        )

        observation = Observation(
            robot=state,
            cameras=frames,
        )

        self.assertEqual(len(observation.cameras.frames), 3)
        self.assertIn("wrist", observation.cameras.frames)

    def test_action(self):
        action = Action(
            q=np.ones(6),
        )

        self.assertEqual(action.q.shape, (6,))

    def test_detection_result(self):
        result = DetectionResult(
            success=True,
            score=0.95,
            consecutive=12,
            reason="depth_confirmed",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.consecutive, 12)


if __name__ == "__main__":
    unittest.main()