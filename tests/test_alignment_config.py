import tempfile
import unittest
from pathlib import Path

from usb_insertion.vision.alignment.config import (
    load_vision_alignment_config,
)


class TestAlignmentConfig(
    unittest.TestCase
):
    def test_load_config(self):
        content = """
correct_face_label: marked

cameras:
  top:
    roi:
      x: 10
      y: 20
      width: 200
      height: 100
    template_path: top.png
    template_tip_xy: [20, 30]
    template_axis_xy: [80, 30]
    target_tip_xy: [200, 200]
    target_axis_angle_deg: 0
    min_matches: 8
    ratio_test: 0.75
    ransac_reproj_threshold: 3
    capture_tip_distance_px: 120
    capture_angle_deg: 20
    aligned_dx_px: 6
    aligned_dy_px: 6
    aligned_angle_deg: 3

  side:
    roi:
      x: 10
      y: 20
      width: 200
      height: 100
    template_path: side.png
    template_tip_xy: [20, 30]
    template_axis_xy: [80, 30]
    target_tip_xy: [200, 200]
    target_axis_angle_deg: 0
    min_matches: 8
    ratio_test: 0.75
    ransac_reproj_threshold: 3
    capture_tip_distance_px: 120
    capture_angle_deg: 20
    aligned_dx_px: 6
    aligned_dy_px: 6
    aligned_angle_deg: 3

servo:
  error_keys:
    - top_dx_px
    - top_dy_px
    - top_angle_deg
    - side_dx_px
    - side_dy_px
    - side_angle_deg

  controlled_joint_indices:
    - 0
    - 1
    - 2
    - 3
    - 4

  jacobian_path: jacobian.npy

  gain: 0.35
  damping: 0.1

  max_joint_delta:
    - 0.4
    - 0.4
    - 0.4
    - 0.4
    - 0.4
"""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            config_dir = (
                root
                / "configs"
            )

            config_dir.mkdir()

            path = (
                config_dir
                / "vision.yaml"
            )

            path.write_text(
                content,
                encoding="utf-8",
            )

            config = (
                load_vision_alignment_config(
                    path
                )
            )

            self.assertEqual(
                set(
                    config.cameras
                ),
                {
                    "top",
                    "side",
                },
            )

            self.assertEqual(
                config.correct_face_label,
                "marked",
            )

            self.assertEqual(
                len(
                    config
                    .servo
                    .error_keys
                ),
                6,
            )


if __name__ == "__main__":
    unittest.main()