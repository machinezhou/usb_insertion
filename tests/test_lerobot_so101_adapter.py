import unittest

import numpy as np

from usb_insertion.adapters.robot.lerobot_so101 import (
    LeRobotSO101Adapter,
    SO101_CAMERA_KEYS,
    SO101_JOINT_KEYS,
)
from usb_insertion.core.datatypes import Action


class StubLeRobotSO101:
    """
    模拟 LeRobot SO101Follower 的最小接口。

    不是 FakeRobot。

    它模拟的是“LeRobot 原始 API”，
    用于测试 Adapter。
    """

    def __init__(self):
        self.connected = False
        self.last_calibrate = None
        self.last_action = None

    def connect(
        self,
        calibrate: bool = True,
    ) -> None:
        self.connected = True
        self.last_calibrate = calibrate

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self):
        return {
            "shoulder_pan.pos": -2.0,
            "shoulder_lift.pos": -80.0,
            "elbow_flex.pos": 96.0,
            "wrist_flex.pos": 65.0,
            "wrist_roll.pos": -17.0,
            "gripper.pos": 10.0,

            "top": np.zeros(
                (480, 640, 3),
                dtype=np.uint8,
            ),

            "wrist": np.zeros(
                (480, 640, 3),
                dtype=np.uint8,
            ),

            "side": np.zeros(
                (480, 640, 3),
                dtype=np.uint8,
            ),
        }

    def send_action(self, action):
        self.last_action = action
        return action


class TestLeRobotSO101Adapter(
    unittest.TestCase
):
    def setUp(self):
        self.raw_robot = (
            StubLeRobotSO101()
        )

        self.adapter = (
            LeRobotSO101Adapter(
                robot=self.raw_robot
            )
        )

    def test_joint_keys_order(self):
        self.assertEqual(
            SO101_JOINT_KEYS,
            (
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ),
        )

    def test_camera_keys(self):
        self.assertEqual(
            SO101_CAMERA_KEYS,
            (
                "top",
                "wrist",
                "side",
            ),
        )

    def test_connect_does_not_calibrate(
        self,
    ):
        self.adapter.connect()

        self.assertTrue(
            self.raw_robot.connected
        )

        self.assertFalse(
            self.raw_robot.last_calibrate
        )

    def test_close_disconnects(self):
        self.adapter.connect()
        self.adapter.close()

        self.assertFalse(
            self.raw_robot.connected
        )

    def test_get_observation_joint_order(
        self,
    ):
        observation = (
            self.adapter.get_observation()
        )

        expected = np.array([
            -2.0,
            -80.0,
            96.0,
            65.0,
            -17.0,
            10.0,
        ])

        np.testing.assert_allclose(
            observation.robot.q,
            expected,
        )

    def test_get_observation_cameras(
        self,
    ):
        observation = (
            self.adapter.get_observation()
        )

        self.assertEqual(
            set(
                observation
                .cameras
                .frames
                .keys()
            ),
            {
                "top",
                "wrist",
                "side",
            },
        )

        for frame in (
            observation
            .cameras
            .frames
            .values()
        ):
            self.assertEqual(
                frame.shape,
                (480, 640, 3),
            )

    def test_send_action_mapping(self):
        action = Action(
            q=np.array([
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ])
        )

        self.adapter.send_action(
            action
        )

        self.assertEqual(
            self.raw_robot.last_action,
            {
                "shoulder_pan.pos": 1.0,
                "shoulder_lift.pos": 2.0,
                "elbow_flex.pos": 3.0,
                "wrist_flex.pos": 4.0,
                "wrist_roll.pos": 5.0,
                "gripper.pos": 6.0,
            },
        )

    def test_wrong_action_shape_fails(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            self.adapter.send_action(
                Action(
                    q=np.zeros(5)
                )
            )

    def test_missing_joint_key_fails(
        self,
    ):
        def bad_observation():
            raw = (
                StubLeRobotSO101()
                .get_observation()
            )

            del raw[
                "elbow_flex.pos"
            ]

            return raw

        self.raw_robot.get_observation = (
            bad_observation
        )

        with self.assertRaises(
            KeyError
        ):
            self.adapter.get_observation()


if __name__ == "__main__":
    unittest.main()