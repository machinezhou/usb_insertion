import unittest

import numpy as np

from usb_insertion.core.protocols import (
    CameraBackend,
    PolicyBackend,
    RobotBackend,
    SuccessDetector,
)
from usb_insertion.core.datatypes import (
    Action,
    DetectionResult,
    FrameBundle,
    Observation,
    RobotState,
)


class DummyRobot:
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_state(self) -> RobotState:
        return RobotState(
            q=np.zeros(6),
            timestamp_s=0.0,
        )

    def send_action(self, action: Action) -> None:
        pass


class DummyCamera:
    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self) -> FrameBundle:
        return FrameBundle(
            frames={},
            timestamp_s=0.0,
        )


class DummyPolicy:
    def reset(self) -> None:
        pass

    def predict(self, observation: Observation) -> Action:
        return Action(
            q=np.zeros(6),
        )


class DummyDetector:
    def reset(self) -> None:
        pass

    def update(self, frames: FrameBundle) -> DetectionResult:
        return DetectionResult(
            success=False,
        )


class TestInterfaces(unittest.TestCase):
    def test_dummy_robot_methods(self):
        robot = DummyRobot()

        robot.connect()
        state = robot.get_state()

        self.assertEqual(state.q.shape, (6,))

        robot.send_action(
            Action(q=np.zeros(6))
        )

        robot.close()

    def test_dummy_camera_methods(self):
        camera = DummyCamera()

        camera.open()
        frames = camera.read()

        self.assertEqual(frames.timestamp_s, 0.0)

        camera.close()

    def test_dummy_policy_methods(self):
        policy = DummyPolicy()

        observation = Observation(
            robot=RobotState(
                q=np.zeros(6),
                timestamp_s=0.0,
            ),
            cameras=FrameBundle(
                frames={},
                timestamp_s=0.0,
            ),
        )

        action = policy.predict(observation)

        self.assertEqual(action.q.shape, (6,))

    def test_dummy_detector_methods(self):
        detector = DummyDetector()

        result = detector.update(
            FrameBundle(
                frames={},
                timestamp_s=0.0,
            )
        )

        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()