import tempfile
import unittest
from pathlib import Path

from usb_insertion.config.hardware import (
    load_hardware_config,
)


class TestHardwareConfig(unittest.TestCase):
    def test_load_project_hardware_config(self):
        project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        config = load_hardware_config(
            project_root
            / "configs"
            / "hardware.yaml"
        )

        self.assertEqual(
            config.robot.type,
            "so101_follower",
        )

        self.assertEqual(
            config.robot.port,
            "/dev/ttyACM0",
        )

        self.assertEqual(
            config.robot.id,
            "lawson_follower_arm",
        )

        self.assertFalse(
            config.robot.calibrate_on_connect
        )

        self.assertEqual(
            set(config.cameras),
            {
                "top",
                "wrist",
                "side",
            },
        )

        self.assertEqual(
            config.cameras["top"].index_or_path,
            2,
        )

        self.assertEqual(
            config.cameras["top"].fourcc,
            "MJPG",
        )

        self.assertEqual(
            config.cameras["wrist"].index_or_path,
            0,
        )

        self.assertEqual(
            config.cameras["side"].index_or_path,
            4,
        )

    def test_invalid_camera_set_fails(self):
        content = """
robot:
  type: so101_follower
  port: /dev/ttyACM0
  id: test
  use_degrees: true
  disable_torque_on_disconnect: true
  calibrate_on_connect: false

cameras:
  wrist:
    type: opencv
    index_or_path: 0
    width: 640
    height: 480
    fps: 30
    fourcc: YUYV
"""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hardware.yaml"

            path.write_text(
                content,
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_hardware_config(path)


if __name__ == "__main__":
    unittest.main()