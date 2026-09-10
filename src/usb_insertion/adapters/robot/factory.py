from __future__ import annotations

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)

from usb_insertion.adapters.robot.lerobot_so101 import (
    LeRobotSO101Adapter,
)
from usb_insertion.config.hardware import (
    HardwareConfig,
)


def build_camera_configs(
    hardware: HardwareConfig,
) -> dict[str, OpenCVCameraConfig]:
    """
    将项目自己的 CameraConfig
    转换为 LeRobot OpenCVCameraConfig。

    这里只构造配置对象，不打开相机。
    """

    cameras: dict[str, OpenCVCameraConfig] = {}

    for name, camera in hardware.cameras.items():
        cameras[name] = OpenCVCameraConfig(
            index_or_path=camera.index_or_path,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            fourcc=camera.fourcc,
        )

    return cameras


def build_so101_follower(
    hardware: HardwareConfig,
) -> SO101Follower:
    """
    根据 HardwareConfig 创建 LeRobot SO101Follower。

    注意：
    这里只构造对象，不调用 connect()。
    """

    cameras = build_camera_configs(
        hardware
    )

    config = SO101FollowerConfig(
        port=hardware.robot.port,
        id=hardware.robot.id,
        cameras=cameras,
        use_degrees=hardware.robot.use_degrees,
        disable_torque_on_disconnect=(
            hardware.robot
            .disable_torque_on_disconnect
        ),
    )

    return SO101Follower(
        config
    )


def build_so101_adapter(
    hardware: HardwareConfig,
) -> LeRobotSO101Adapter:
    """
    创建项目使用的 SO-101 Adapter。

    内部仍然使用 LeRobot 原生 SO101Follower。
    """

    raw_robot = build_so101_follower(
        hardware
    )

    return LeRobotSO101Adapter(
        robot=raw_robot,
        calibrate_on_connect=(
            hardware.robot
            .calibrate_on_connect
        ),
    )