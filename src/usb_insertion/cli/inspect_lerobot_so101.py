from __future__ import annotations

from typing import Any

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)


def describe_value(
    key: str,
    value: Any,
) -> None:
    """
    只打印 observation value 的元信息。

    图像：
        打印 shape / dtype / min / max

    标量：
        打印类型和值
    """

    if isinstance(value, np.ndarray):
        print(
            f"{key}: "
            f"type=ndarray, "
            f"shape={value.shape}, "
            f"dtype={value.dtype}, "
            f"min={value.min()}, "
            f"max={value.max()}"
        )
        return

    print(
        f"{key}: "
        f"type={type(value).__name__}, "
        f"value={value}"
    )


def main() -> None:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=2,
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=0,
            width=640,
            height=480,
            fps=30,
            fourcc="YUYV",
        ),
        "side": OpenCVCameraConfig(
            index_or_path=4,
            width=640,
            height=480,
            fps=30,
            fourcc="YUYV",
        ),
    }

    config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="lawson_follower_arm",
        cameras=cameras,
    )

    robot = SO101Follower(
        config
    )

    print("===================================")
    print("SO-101 READ-ONLY INSPECTION")
    print("===================================")

    print()
    print("Robot class:")
    print(type(robot))

    print()
    print("Connecting...")
    print("NOTE: calibrate=False")
    print("NOTE: this script never calls send_action()")

    connected = False

    try:
        # 这里只检查现有机器人。
        # 不允许本探针主动启动 calibration。
        robot.connect(
            calibrate=False
        )

        connected = True

        print()
        print("Connected.")

        # ---------------------------------------------
        # Features
        # ---------------------------------------------

        print()
        print("========== OBSERVATION FEATURES ==========")

        observation_features = getattr(
            robot,
            "observation_features",
            None,
        )

        print(
            observation_features
        )

        print()
        print("============ ACTION FEATURES =============")

        action_features = getattr(
            robot,
            "action_features",
            None,
        )

        print(
            action_features
        )

        # ---------------------------------------------
        # Read exactly one observation
        # ---------------------------------------------

        print()
        print("========== GET OBSERVATION ===============")

        observation = (
            robot.get_observation()
        )

        print()
        print(
            "Observation type:",
            type(observation),
        )

        print(
            "Observation keys:"
        )

        for key in observation.keys():
            print(
                f"  - {key}"
            )

        print()
        print("========== VALUES =========================")

        for key, value in observation.items():
            describe_value(
                key,
                value,
            )

        print()
        print("===================================")
        print("READ-ONLY INSPECTION FINISHED")
        print("===================================")

    finally:
        if connected:
            print()
            print("Disconnecting...")

            robot.disconnect()

            print("Disconnected.")


if __name__ == "__main__":
    main()