import unittest

from usb_insertion.fakes.fake_cameras import (
    FakeCameras,
)
from usb_insertion.fakes.fake_observation_source import (
    FakeObservationSource,
)
from usb_insertion.fakes.fake_robot import (
    FakeRobot,
)


class TestFakeObservationSource(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot(dof=6)
        self.cameras = FakeCameras()

        self.robot.connect()
        self.cameras.open()

        self.source = FakeObservationSource(
            robot=self.robot,
            cameras=self.cameras,
        )

    def tearDown(self):
        self.cameras.close()
        self.robot.close()

    def test_get_observation(self):
        observation = (
            self.source.get_observation()
        )

        self.assertEqual(
            observation.robot.q.shape,
            (6,),
        )

        self.assertEqual(
            set(
                observation.cameras.frames.keys()
            ),
            {
                "top",
                "wrist",
                "side",
            },
        )

    def test_camera_shapes(self):
        observation = (
            self.source.get_observation()
        )

        self.assertEqual(
            observation.cameras.frames[
                "top"
            ].shape,
            (480, 640, 3),
        )

        self.assertEqual(
            observation.cameras.frames[
                "wrist"
            ].shape,
            (480, 640, 3),
        )

        self.assertEqual(
            observation.cameras.frames[
                "side"
            ].shape,
            (480, 640, 3),
        )


if __name__ == "__main__":
    unittest.main()