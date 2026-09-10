import tempfile
import unittest
from pathlib import Path

from usb_insertion.config.run import (
    load_run_config,
)


VALID = """
control_fps: 30
home_q: [0, -90, 90, 60, 0, 10]
home_duration_s: 4.0
max_attempts: 2
visual_alignment_timeout_s: 8.0
max_delta_per_step: [1, 1, 1, 1, 1, 1]

act:
  checkpoint: checkpoints/act
  dataset_repo_id: lawson/usb_pick_approach
  dataset_root: datasets/usb_pick_approach
  task: Pick up the USB plug and move it near the socket
  device: cuda
  timeout_s: 12.0

insertion:
  joint_delta_path: outputs/insertion/insertion_joint_delta.npy
  success_reference_path: outputs/insertion/success_reference.yaml
  progress_step: 0.02
  timeout_s: 8.0

  correction_error_keys:
    - top_dy_px
    - top_angle_deg
    - side_dy_px
    - side_angle_deg

  correction_gain: 0.25
  correction_damping: 0.1

  correction_max_joint_delta:
    - 0.2
    - 0.2
    - 0.2
    - 0.2
    - 0.2

  guard_error_limits:
    top_dy_px: 10
    top_angle_deg: 5
    side_dy_px: 10
    side_angle_deg: 5

  retract_scale: 0.5
  retract_duration_s: 2.0
"""


class TestRunConfig(
    unittest.TestCase
):
    def _load(
        self,
        content: str,
    ):
        tmp = (
            tempfile
            .TemporaryDirectory()
        )

        self.addCleanup(
            tmp.cleanup
        )

        root = Path(
            tmp.name
        )

        config_dir = (
            root
            / "configs"
        )

        config_dir.mkdir()

        path = (
            config_dir
            / "run.yaml"
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        return (
            load_run_config(
                path
            ),
            root,
        )

    def test_valid_config(self):
        config, root = (
            self._load(
                VALID
            )
        )

        self.assertEqual(
            config.control_fps,
            30.0,
        )

        self.assertEqual(
            len(
                config.home_q
            ),
            6,
        )

        self.assertEqual(
            config.max_attempts,
            2,
        )

        self.assertEqual(
            config.act.device,
            "cuda",
        )

        self.assertEqual(
            config.act.dataset_root,
            (
                root
                / "datasets"
                / "usb_pick_approach"
            ),
        )

        self.assertEqual(
            (
                config
                .insertion
                .joint_delta_path
            ),
            (
                root
                / "outputs"
                / "insertion"
                / "insertion_joint_delta.npy"
            ),
        )

    def test_bad_progress_fails(self):
        content = VALID.replace(
            "progress_step: 0.02",
            "progress_step: 1.5",
        )

        with self.assertRaises(
            ValueError
        ):
            self._load(
                content
            )

    def test_bad_action_delta_fails(
        self,
    ):
        content = VALID.replace(
            (
                "max_delta_per_step: "
                "[1, 1, 1, 1, 1, 1]"
            ),
            (
                "max_delta_per_step: "
                "[1, 1, 0, 1, 1, 1]"
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            self._load(
                content
            )


if __name__ == "__main__":
    unittest.main()
