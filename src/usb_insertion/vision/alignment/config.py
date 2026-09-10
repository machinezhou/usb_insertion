from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from usb_insertion.vision.roi import ImageROI


@dataclass(frozen=True, slots=True)
class CameraAlignmentConfig:
    camera_name: str

    roi: ImageROI

    template_path: Path

    # 模板图内部坐标
    template_tip_xy: tuple[float, float]
    template_axis_xy: tuple[float, float]

    # 理想 Pre-insert 时 USB tip 在整张图中的位置
    target_tip_xy: tuple[float, float]

    # 理想 Pre-insert 时 USB 轴线角度
    target_axis_angle_deg: float

    # Feature matching
    min_matches: int
    ratio_test: float
    ransac_reproj_threshold: float

    # ACT 是否已经进入 Visual Servo 捕获区域
    capture_tip_distance_px: float
    capture_angle_deg: float

    # Visual Servo 是否已经完成精确对准
    aligned_dx_px: float
    aligned_dy_px: float
    aligned_angle_deg: float


@dataclass(frozen=True, slots=True)
class VisualServoConfig:
    error_keys: tuple[str, ...]

    controlled_joint_indices: tuple[int, ...]

    jacobian_path: Path

    gain: float
    damping: float

    max_joint_delta: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VisionAlignmentConfig:
    correct_face_label: str

    cameras: dict[
        str,
        CameraAlignmentConfig,
    ]

    servo: VisualServoConfig


def _xy(
    value: object,
    name: str,
) -> tuple[float, float]:
    if not isinstance(
        value,
        (list, tuple),
    ):
        raise ValueError(
            f"{name} must be [x, y]"
        )

    if len(value) != 2:
        raise ValueError(
            f"{name} must contain 2 values"
        )

    return (
        float(value[0]),
        float(value[1]),
    )


def load_vision_alignment_config(
    path: str | Path,
) -> VisionAlignmentConfig:

    path = Path(path)

    project_root = (
        path.resolve().parents[1]
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(
            "vision config root must be a mapping"
        )

    correct_face_label = str(
        raw.get(
            "correct_face_label",
            "marked",
        )
    )

    cameras_raw = raw.get(
        "cameras"
    )

    if not isinstance(
        cameras_raw,
        dict,
    ):
        raise ValueError(
            "'cameras' must be a mapping"
        )

    cameras: dict[
        str,
        CameraAlignmentConfig,
    ] = {}

    for camera_name, camera_raw in (
        cameras_raw.items()
    ):
        if not isinstance(
            camera_raw,
            dict,
        ):
            raise ValueError(
                f"camera '{camera_name}' "
                "must be a mapping"
            )

        roi_raw = camera_raw[
            "roi"
        ]

        roi = ImageROI(
            x=int(roi_raw["x"]),
            y=int(roi_raw["y"]),
            width=int(
                roi_raw["width"]
            ),
            height=int(
                roi_raw["height"]
            ),
        )

        template_path = Path(
            camera_raw[
                "template_path"
            ]
        )

        if not template_path.is_absolute():
            template_path = (
                project_root
                / template_path
            )

        min_matches = int(
            camera_raw.get(
                "min_matches",
                8,
            )
        )

        ratio_test = float(
            camera_raw.get(
                "ratio_test",
                0.75,
            )
        )

        ransac = float(
            camera_raw.get(
                "ransac_reproj_threshold",
                3.0,
            )
        )

        if min_matches < 4:
            raise ValueError(
                "min_matches must be >= 4"
            )

        if not 0 < ratio_test < 1:
            raise ValueError(
                "ratio_test must be in (0, 1)"
            )

        cameras[camera_name] = (
            CameraAlignmentConfig(
                camera_name=camera_name,
                roi=roi,
                template_path=(
                    template_path
                ),
                template_tip_xy=_xy(
                    camera_raw[
                        "template_tip_xy"
                    ],
                    (
                        f"{camera_name}."
                        "template_tip_xy"
                    ),
                ),
                template_axis_xy=_xy(
                    camera_raw[
                        "template_axis_xy"
                    ],
                    (
                        f"{camera_name}."
                        "template_axis_xy"
                    ),
                ),
                target_tip_xy=_xy(
                    camera_raw[
                        "target_tip_xy"
                    ],
                    (
                        f"{camera_name}."
                        "target_tip_xy"
                    ),
                ),
                target_axis_angle_deg=float(
                    camera_raw[
                        "target_axis_angle_deg"
                    ]
                ),
                min_matches=min_matches,
                ratio_test=ratio_test,
                ransac_reproj_threshold=(
                    ransac
                ),
                capture_tip_distance_px=float(
                    camera_raw[
                        "capture_tip_distance_px"
                    ]
                ),
                capture_angle_deg=float(
                    camera_raw[
                        "capture_angle_deg"
                    ]
                ),
                aligned_dx_px=float(
                    camera_raw[
                        "aligned_dx_px"
                    ]
                ),
                aligned_dy_px=float(
                    camera_raw[
                        "aligned_dy_px"
                    ]
                ),
                aligned_angle_deg=float(
                    camera_raw[
                        "aligned_angle_deg"
                    ]
                ),
            )
        )

    required = {
        "top",
        "side",
    }

    if set(cameras) != required:
        raise ValueError(
            "Vision Alignment V1 requires "
            "exactly top and side cameras"
        )

    servo_raw = raw.get(
        "servo"
    )

    if not isinstance(
        servo_raw,
        dict,
    ):
        raise ValueError(
            "'servo' must be a mapping"
        )

    jacobian_path = Path(
        servo_raw[
            "jacobian_path"
        ]
    )

    if not (
        jacobian_path.is_absolute()
    ):
        jacobian_path = (
            project_root
            / jacobian_path
        )

    indices = tuple(
        int(value)
        for value in servo_raw[
            "controlled_joint_indices"
        ]
    )

    max_joint_delta = tuple(
        float(value)
        for value in servo_raw[
            "max_joint_delta"
        ]
    )

    if (
        len(max_joint_delta)
        != len(indices)
    ):
        raise ValueError(
            "max_joint_delta and "
            "controlled_joint_indices "
            "must have same length"
        )

    servo = VisualServoConfig(
        error_keys=tuple(
            str(value)
            for value in servo_raw[
                "error_keys"
            ]
        ),
        controlled_joint_indices=(
            indices
        ),
        jacobian_path=(
            jacobian_path
        ),
        gain=float(
            servo_raw.get(
                "gain",
                0.35,
            )
        ),
        damping=float(
            servo_raw.get(
                "damping",
                0.1,
            )
        ),
        max_joint_delta=(
            max_joint_delta
        ),
    )

    return VisionAlignmentConfig(
        correct_face_label=(
            correct_face_label
        ),
        cameras=cameras,
        servo=servo,
    )