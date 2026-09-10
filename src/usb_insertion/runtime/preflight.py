from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from usb_insertion.config.hardware import (
    HardwareConfig,
)
from usb_insertion.config.run import (
    RunConfig,
)
from usb_insertion.vision.alignment.config import (
    VisionAlignmentConfig,
)
from usb_insertion.vision.alignment.face import (
    PlugFaceConfig,
)
from usb_insertion.vision.alignment.jacobian import (
    load_image_jacobian,
)
from usb_insertion.vision.insertion.success_detector import (
    load_success_reference,
)


@dataclass(
    frozen=True,
    slots=True,
)
class PreflightCheck:
    name: str
    passed: bool
    detail: str


@dataclass(
    frozen=True,
    slots=True,
)
class PreflightReport:
    checks: tuple[
        PreflightCheck,
        ...,
    ]

    @property
    def passed(self) -> bool:
        return all(
            check.passed
            for check in self.checks
        )

    def format_text(self) -> str:
        lines = []

        for check in self.checks:
            marker = (
                "PASS"
                if check.passed
                else "FAIL"
            )

            lines.append(
                f"[{marker}] "
                f"{check.name}: "
                f"{check.detail}"
            )

        return "\n".join(
            lines
        )

    def raise_if_failed(
        self,
    ) -> None:

        if self.passed:
            return

        failures = [
            check
            for check
            in self.checks
            if not check.passed
        ]

        message = "\n".join(
            f"- {check.name}: "
            f"{check.detail}"
            for check in failures
        )

        raise RuntimeError(
            "USB insertion "
            "preflight failed:\n"
            + message
        )


def _finite_vector(
    values,
    expected_length: int,
) -> bool:

    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except Exception:
        return False

    return (
        array.shape
        == (
            expected_length,
        )
        and bool(
            np.all(
                np.isfinite(
                    array
                )
            )
        )
    )


def run_preflight(
    *,
    hardware: HardwareConfig,
    vision: VisionAlignmentConfig,
    face: PlugFaceConfig,
    run_cfg: RunConfig,
    check_devices: bool = False,
) -> PreflightReport:

    checks = []

    def add(
        name,
        passed,
        detail,
    ):
        checks.append(
            PreflightCheck(
                name=name,
                passed=bool(
                    passed
                ),
                detail=detail,
            )
        )

    add(
        "robot_use_degrees",
        hardware
        .robot
        .use_degrees,
        (
            "control parameters "
            "expect degree space"
        ),
    )

    add(
        "camera_set",
        set(
            hardware.cameras
        )
        == {
            "top",
            "wrist",
            "side",
        },
        str(
            tuple(
                hardware.cameras
            )
        ),
    )

    min_camera_fps = min(
        camera.fps
        for camera
        in hardware
        .cameras
        .values()
    )

    add(
        "control_fps_vs_camera",
        (
            0
            < run_cfg.control_fps
            <= min_camera_fps
        ),
        (
            f"control="
            f"{run_cfg.control_fps}, "
            f"camera="
            f"{min_camera_fps}"
        ),
    )

    add(
        "home_q",
        _finite_vector(
            run_cfg.home_q,
            6,
        ),
        str(
            run_cfg.home_q
        ),
    )

    max_delta_ok = (
        _finite_vector(
            run_cfg
            .max_delta_per_step,
            6,
        )
        and all(
            value > 0
            for value
            in run_cfg
            .max_delta_per_step
        )
    )

    add(
        "action_guard_delta",
        max_delta_ok,
        str(
            run_cfg
            .max_delta_per_step
        ),
    )

    for (
        camera_name,
        camera_cfg,
    ) in vision.cameras.items():

        add(
            (
                "vision_template:"
                f"{camera_name}"
            ),
            camera_cfg
            .template_path
            .is_file(),
            str(
                camera_cfg
                .template_path
            ),
        )

        hw_camera = (
            hardware
            .cameras
            .get(
                camera_name
            )
        )

        roi_ok = (
            hw_camera is not None
            and camera_cfg.roi.x2
            <= hw_camera.width
            and camera_cfg.roi.y2
            <= hw_camera.height
        )

        add(
            (
                "vision_roi:"
                f"{camera_name}"
            ),
            roi_ok,
            str(
                camera_cfg.roi
            ),
        )

    add(
        "face_camera",
        face.camera_name
        in hardware.cameras,
        face.camera_name,
    )

    face_hw_camera = (
        hardware
        .cameras
        .get(
            face.camera_name
        )
    )

    face_roi_ok = (
        face_hw_camera is not None
        and face.roi.x2
        <= face_hw_camera.width
        and face.roi.y2
        <= face_hw_camera.height
    )

    add(
        "face_roi",
        face_roi_ok,
        str(face.roi),
    )

    add(
        "face_correct_label",
        face.correct_label
        in face.templates,
        face.correct_label,
    )

    add(
        "vision_face_label_consistency",
        (
            vision
            .correct_face_label
            == face.correct_label
        ),
        (
            f"vision="
            f"{vision.correct_face_label}, "
            f"face="
            f"{face.correct_label}"
        ),
    )

    for label, path in (
        face.templates.items()
    ):
        add(
            f"face_template:{label}",
            path.is_file(),
            str(path),
        )

    servo_indices = (
        vision.servo
        .controlled_joint_indices
    )

    add(
        "visual_servo_joint_indices",
        (
            bool(
                servo_indices
            )
            and len(
                set(
                    servo_indices
                )
            )
            == len(
                servo_indices
            )
            and all(
                0 <= index < 6
                for index
                in servo_indices
            )
        ),
        str(
            servo_indices
        ),
    )

    add(
        "visual_servo_no_gripper",
        5 not in servo_indices,
        str(
            servo_indices
        ),
    )

    try:
        model = (
            load_image_jacobian(
                vision.servo
                .jacobian_path,
                error_keys=(
                    vision.servo
                    .error_keys
                ),
                controlled_joint_indices=(
                    servo_indices
                ),
            )
        )

        jacobian_ok = True

        jacobian_detail = (
            f"{model.shape}"
        )

    except Exception as exc:
        jacobian_ok = False
        jacobian_detail = str(
            exc
        )

    add(
        "image_jacobian",
        jacobian_ok,
        jacobian_detail,
    )

    try:
        delta = np.asarray(
            np.load(
                run_cfg
                .insertion
                .joint_delta_path,
                allow_pickle=False,
            ),
            dtype=np.float64,
        )

        delta_ok = (
            delta.shape == (6,)
            and np.all(
                np.isfinite(
                    delta
                )
            )
            and abs(
                float(
                    delta[5]
                )
            )
            <= 1e-9
            and np.linalg.norm(
                delta[:5]
            ) > 0
        )

        delta_detail = str(
            delta
        )

    except Exception as exc:
        delta_ok = False
        delta_detail = str(
            exc
        )

    add(
        "insertion_joint_delta",
        delta_ok,
        delta_detail,
    )

    vision_error_keys = {
        (
            f"{camera_name}_"
            f"{suffix}"
        )
        for camera_name
        in vision.cameras
        for suffix in (
            "dx_px",
            "dy_px",
            "angle_deg",
        )
    }

    correction_keys = set(
        run_cfg
        .insertion
        .correction_error_keys
    )

    guard_keys = set(
        run_cfg
        .insertion
        .guard_error_limits
    )

    add(
        "insertion_correction_errors",
        (
            bool(
                correction_keys
            )
            and correction_keys
            <= vision_error_keys
        ),
        str(
            correction_keys
        ),
    )

    add(
        "insertion_guard_errors",
        (
            bool(
                guard_keys
            )
            and guard_keys
            <= vision_error_keys
        ),
        str(
            guard_keys
        ),
    )

    add(
        (
            "insertion_correction_"
            "delta_shape"
        ),
        (
            len(
                run_cfg
                .insertion
                .correction_max_joint_delta
            )
            == len(
                servo_indices
            )
        ),
        (
            f"{len(run_cfg.insertion.correction_max_joint_delta)} "
            f"vs {len(servo_indices)}"
        ),
    )

    try:
        success_reference = (
            load_success_reference(
                run_cfg
                .insertion
                .success_reference_path
            )
        )

        success_keys = set(
            success_reference
            .target_errors
        )

        success_ok = (
            bool(
                success_keys
            )
            and success_keys
            <= vision_error_keys
        )

        success_detail = str(
            success_keys
        )

    except Exception as exc:
        success_ok = False
        success_detail = str(
            exc
        )

    add(
        "success_reference",
        success_ok,
        success_detail,
    )

    if (
        run_cfg.act
        .dataset_root
        is not None
    ):
        add(
            "act_dataset_root",
            run_cfg.act
            .dataset_root
            .exists(),
            str(
                run_cfg.act
                .dataset_root
            ),
        )

    else:
        add(
            "act_dataset_root",
            True,
            (
                "not set; LeRobot/HF "
                "cache may be used"
            ),
        )

    add(
        "act_checkpoint",
        bool(
            run_cfg.act
            .checkpoint
            .strip()
        ),
        run_cfg.act.checkpoint,
    )

    add(
        "act_dataset_repo_id",
        bool(
            run_cfg.act
            .dataset_repo_id
            .strip()
        ),
        run_cfg
        .act
        .dataset_repo_id,
    )

    add(
        "act_task",
        bool(
            run_cfg.act
            .task
            .strip()
        ),
        run_cfg.act.task,
    )

    if check_devices:
        robot_port = Path(
            hardware.robot.port
        )

        add(
            "device:robot",
            robot_port.exists(),
            str(robot_port),
        )

        for (
            camera_name,
            camera,
        ) in (
            hardware
            .cameras
            .items()
        ):
            if isinstance(
                camera.index_or_path,
                int,
            ):
                device = Path(
                    "/dev/"
                    f"video"
                    f"{camera.index_or_path}"
                )

            else:
                device = Path(
                    str(
                        camera
                        .index_or_path
                    )
                )

            add(
                (
                    "device:camera:"
                    f"{camera_name}"
                ),
                device.exists(),
                str(device),
            )

    return PreflightReport(
        checks=tuple(
            checks
        )
    )
