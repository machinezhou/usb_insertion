import tempfile
import unittest
from pathlib import Path

import numpy as np

from usb_insertion.config.hardware import (
    CameraConfig,
    HardwareConfig,
    RobotConfig,
)
from usb_insertion.config.run import (
    ACTRunConfig,
    InsertionRunConfig,
    RunConfig,
)
from usb_insertion.runtime.preflight import (
    run_preflight,
)
from usb_insertion.vision.alignment.config import (
    CameraAlignmentConfig,
    VisionAlignmentConfig,
    VisualServoConfig,
)
from usb_insertion.vision.alignment.face import (
    PlugFaceConfig,
)
from usb_insertion.vision.roi import (
    ImageROI,
)


class TestPreflight(
    unittest.TestCase
):
    def _objects(
        self,
        root: Path,
    ):
        hardware = HardwareConfig(
            robot=RobotConfig(
                type="so101_follower",
                port="/dev/ttyACM0",
                id="test",
                use_degrees=True,
                disable_torque_on_disconnect=True,
                calibrate_on_connect=False,
            ),
            cameras={
                name: CameraConfig(
                    type="opencv",
                    index_or_path=index,
                    width=640,
                    height=480,
                    fps=30,
                    fourcc="YUYV",
                )
                for name, index in (
                    (
                        "top",
                        2,
                    ),
                    (
                        "wrist",
                        0,
                    ),
                    (
                        "side",
                        4,
                    ),
                )
            },
        )

        top_template = (
            root
            / "top.png"
        )

        side_template = (
            root
            / "side.png"
        )

        marked = (
            root
            / "marked.png"
        )

        unmarked = (
            root
            / "unmarked.png"
        )

        for path in (
            top_template,
            side_template,
            marked,
            unmarked,
        ):
            path.write_bytes(
                b"x"
            )

        jacobian = (
            root
            / "jacobian.npy"
        )

        np.save(
            jacobian,
            np.ones(
                (
                    6,
                    5,
                ),
                dtype=np.float64,
            ),
        )

        delta = (
            root
            / "delta.npy"
        )

        np.save(
            delta,
            np.array(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    0,
                ],
                dtype=float,
            ),
        )

        success = (
            root
            / "success.yaml"
        )

        success.write_text(
            """
target_errors:
  top_dy_px: 1.0
  side_dy_px: 2.0

tolerances:
  top_dy_px: 3.0
  side_dy_px: 3.0

required_consecutive: 3
""",
            encoding="utf-8",
        )

        def camera_alignment(
            name,
            template,
        ):
            return (
                CameraAlignmentConfig(
                    camera_name=name,
                    roi=ImageROI(
                        0,
                        0,
                        100,
                        100,
                    ),
                    template_path=(
                        template
                    ),
                    template_tip_xy=(
                        10.0,
                        10.0,
                    ),
                    template_axis_xy=(
                        40.0,
                        10.0,
                    ),
                    target_tip_xy=(
                        100.0,
                        100.0,
                    ),
                    target_axis_angle_deg=(
                        0.0
                    ),
                    min_matches=8,
                    ratio_test=0.75,
                    ransac_reproj_threshold=(
                        3.0
                    ),
                    capture_tip_distance_px=(
                        120.0
                    ),
                    capture_angle_deg=(
                        20.0
                    ),
                    aligned_dx_px=(
                        6.0
                    ),
                    aligned_dy_px=(
                        6.0
                    ),
                    aligned_angle_deg=(
                        3.0
                    ),
                )
            )

        vision = (
            VisionAlignmentConfig(
                correct_face_label=(
                    "marked"
                ),
                cameras={
                    "top": (
                        camera_alignment(
                            "top",
                            top_template,
                        )
                    ),
                    "side": (
                        camera_alignment(
                            "side",
                            side_template,
                        )
                    ),
                },
                servo=(
                    VisualServoConfig(
                        error_keys=(
                            "top_dx_px",
                            "top_dy_px",
                            "top_angle_deg",
                            "side_dx_px",
                            "side_dy_px",
                            "side_angle_deg",
                        ),
                        controlled_joint_indices=(
                            0,
                            1,
                            2,
                            3,
                            4,
                        ),
                        jacobian_path=(
                            jacobian
                        ),
                        gain=0.3,
                        damping=0.1,
                        max_joint_delta=(
                            0.2,
                            0.2,
                            0.2,
                            0.2,
                            0.2,
                        ),
                    )
                ),
            )
        )

        face = PlugFaceConfig(
            camera_name="wrist",
            roi=ImageROI(
                0,
                0,
                100,
                100,
            ),
            templates={
                "marked": marked,
                "unmarked": (
                    unmarked
                ),
            },
            correct_label="marked",
        )

        run_cfg = RunConfig(
            control_fps=30,
            home_q=(
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            home_duration_s=3,
            max_attempts=2,
            visual_alignment_timeout_s=8,
            max_delta_per_step=(
                1,
                1,
                1,
                1,
                1,
                1,
            ),
            act=ACTRunConfig(
                checkpoint=(
                    "lawson/model"
                ),
                dataset_repo_id=(
                    "lawson/data"
                ),
                dataset_root=None,
                task="test",
                device="cpu",
                timeout_s=10,
            ),
            insertion=(
                InsertionRunConfig(
                    joint_delta_path=(
                        delta
                    ),
                    success_reference_path=(
                        success
                    ),
                    progress_step=(
                        0.02
                    ),
                    timeout_s=8,
                    correction_error_keys=(
                        "top_dy_px",
                        "side_dy_px",
                    ),
                    correction_gain=(
                        0.2
                    ),
                    correction_damping=(
                        0.1
                    ),
                    correction_max_joint_delta=(
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                    ),
                    guard_error_limits={
                        "top_dy_px": 10,
                        "side_dy_px": 10,
                    },
                    retract_scale=(
                        0.5
                    ),
                    retract_duration_s=(
                        2
                    ),
                )
            ),
        )

        return (
            hardware,
            vision,
            face,
            run_cfg,
        )

    def test_valid_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            objects = self._objects(
                Path(tmp)
            )

            report = run_preflight(
                hardware=objects[0],
                vision=objects[1],
                face=objects[2],
                run_cfg=objects[3],
                check_devices=False,
            )

            if not report.passed:
                self.fail(
                    report.format_text()
                )

    def test_nonzero_gripper_delta_fails(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            (
                hardware,
                vision,
                face,
                run_cfg,
            ) = self._objects(
                root
            )

            np.save(
                (
                    run_cfg
                    .insertion
                    .joint_delta_path
                ),
                np.array(
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        0.2,
                    ],
                    dtype=float,
                ),
            )

            report = run_preflight(
                hardware=hardware,
                vision=vision,
                face=face,
                run_cfg=run_cfg,
                check_devices=False,
            )

            self.assertFalse(
                report.passed
            )

            self.assertIn(
                "insertion_joint_delta",
                report.format_text(),
            )


if __name__ == "__main__":
    unittest.main()
