from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CameraConfig:
    type: str
    index_or_path: int | str
    width: int
    height: int
    fps: int
    fourcc: str | None


@dataclass(frozen=True, slots=True)
class RobotConfig:
    type: str
    port: str
    id: str

    use_degrees: bool
    disable_torque_on_disconnect: bool
    calibrate_on_connect: bool


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    robot: RobotConfig
    cameras: dict[str, CameraConfig]


def load_hardware_config(
    path: str | Path,
) -> HardwareConfig:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(
            "hardware config root must be a mapping"
        )

    robot_raw = raw.get("robot")
    cameras_raw = raw.get("cameras")

    if not isinstance(robot_raw, dict):
        raise ValueError(
            "'robot' must be a mapping"
        )

    if not isinstance(cameras_raw, dict):
        raise ValueError(
            "'cameras' must be a mapping"
        )

    robot = RobotConfig(
        type=str(robot_raw["type"]),
        port=str(robot_raw["port"]),
        id=str(robot_raw["id"]),
        use_degrees=bool(
            robot_raw["use_degrees"]
        ),
        disable_torque_on_disconnect=bool(
            robot_raw[
                "disable_torque_on_disconnect"
            ]
        ),
        calibrate_on_connect=bool(
            robot_raw[
                "calibrate_on_connect"
            ]
        ),
    )

    cameras = {
        name: CameraConfig(
            type=str(camera_raw["type"]),
            index_or_path=(
                camera_raw["index_or_path"]
            ),
            width=int(camera_raw["width"]),
            height=int(camera_raw["height"]),
            fps=int(camera_raw["fps"]),
            fourcc=(
                None
                if camera_raw.get("fourcc") is None
                else str(
                    camera_raw["fourcc"]
                )
            ),
        )
        for name, camera_raw
        in cameras_raw.items()
    }

    _validate_hardware_config(
        robot=robot,
        cameras=cameras,
    )

    return HardwareConfig(
        robot=robot,
        cameras=cameras,
    )


def _validate_hardware_config(
    robot: RobotConfig,
    cameras: dict[str, CameraConfig],
) -> None:
    if robot.type != "so101_follower":
        raise ValueError(
            "robot.type must be 'so101_follower'"
        )

    if not robot.port:
        raise ValueError(
            "robot.port must not be empty"
        )

    if not robot.id:
        raise ValueError(
            "robot.id must not be empty"
        )

    required_cameras = {
        "top",
        "wrist",
        "side",
    }

    if set(cameras) != required_cameras:
        raise ValueError(
            "cameras must contain exactly: "
            "top, wrist, side"
        )

    for name, camera in cameras.items():
        if camera.type != "opencv":
            raise ValueError(
                f"camera '{name}' type "
                "must be 'opencv'"
            )

        if camera.width <= 0:
            raise ValueError(
                f"camera '{name}' width "
                "must be > 0"
            )

        if camera.height <= 0:
            raise ValueError(
                f"camera '{name}' height "
                "must be > 0"
            )

        if camera.fps <= 0:
            raise ValueError(
                f"camera '{name}' fps "
                "must be > 0"
            )