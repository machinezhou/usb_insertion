import unittest
from pathlib import Path

from lerobot.cameras.opencv import (
    OpenCVCameraConfig,
)

from usb_insertion.adapters.robot.factory import (
    build_camera_configs,
    build_so101_adapter,
    build_so101_follower,
)
from usb_insertion.adapters.robot.lerobot_so101 import (
    LeRobotSO101Adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)


class TestRobotFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        cls.hardware = load_hardware_config(
            project_root
            / "configs"
            / "hardware.yaml"
        )

    def test_build_camera_configs(self):
        cameras = build_camera_configs(
            self.hardware
        )

        self.assertEqual(
            set(cameras),
            {
                "top",
                "wrist",
                "side",
            },
        )

        for camera in cameras.values():
            self.assertIsInstance(
                camera,
                OpenCVCameraConfig,
            )

    def test_camera_values(self):
        cameras = build_camera_configs(
            self.hardware
        )

        self.assertEqual(
            cameras["top"].index_or_path,
            2,
        )

        self.assertEqual(
            cameras["top"].width,
            640,
        )

        self.assertEqual(
            cameras["top"].height,
            480,
        )

        self.assertEqual(
            cameras["top"].fps,
            30,
        )

        self.assertEqual(
            cameras["top"].fourcc,
            "MJPG",
        )

        self.assertEqual(
            cameras["wrist"].fourcc,
            "YUYV",
        )

    def test_build_follower_without_connecting(self):
        robot = build_so101_follower(
            self.hardware
        )

        # 这里只验证能成功构造对象。
        # 不调用 connect()。
        self.assertIsNotNone(
            robot
        )

    def test_build_adapter(self):
        adapter = build_so101_adapter(
            self.hardware
        )

        self.assertIsInstance(
            adapter,
            LeRobotSO101Adapter,
        )

        self.assertIsNotNone(
            adapter.raw_robot
        )


if __name__ == "__main__":
    unittest.main()