from __future__ import annotations

from pathlib import Path

import numpy as np

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.adapters.robot.lerobot_so101 import (
    SO101_CAMERA_KEYS,
    SO101_JOINT_KEYS,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)


def get_project_root() -> Path:
    """
    获取项目根目录。

    当前文件：

        usb_insertion/
        └── src/
            └── usb_insertion/
                └── cli/
                    └── inspect_adapter_so101.py

    parents[3] 对应：

        ~/projects/usb_insertion
    """
    return Path(__file__).resolve().parents[3]


def main() -> None:
    # =========================================================
    # 1. Load project hardware configuration
    # =========================================================

    project_root = get_project_root()

    hardware_config_path = (
        project_root
        / "configs"
        / "hardware.yaml"
    )

    print(
        "========================================"
    )
    print(
        "USB INSERTION - ADAPTER READ-ONLY TEST"
    )
    print(
        "========================================"
    )
    print()

    print(
        "Hardware config:"
    )
    print(
        hardware_config_path
    )
    print()

    hardware = load_hardware_config(
        hardware_config_path
    )

    # =========================================================
    # 2. Build LeRobot SO-101 through project factory
    # =========================================================

    robot = build_so101_adapter(
        hardware
    )

    raw_robot = robot.raw_robot

    print(
        "Robot adapter:"
    )
    print(
        type(robot)
    )

    print()

    print(
        "Raw LeRobot robot:"
    )
    print(
        type(raw_robot)
    )

    print()

    print(
        "This script DOES NOT call send_action()."
    )
    print(
        "This script DOES NOT call "
        "SOFollower.connect()."
    )
    print(
        "SOFollower.configure() is bypassed."
    )
    print(
        "Motor torque state is not intentionally "
        "changed by this inspection."
    )
    print()

    # =========================================================
    # 3. Inspection-only connection state
    # =========================================================

    bus_connected = False

    connected_cameras: list[str] = []

    try:
        # -----------------------------------------------------
        # Motor bus
        # -----------------------------------------------------

        print(
            "Connecting motor bus WITHOUT "
            "robot.configure()..."
        )

        raw_robot.bus.connect()

        bus_connected = True

        # -----------------------------------------------------
        # Cameras
        # -----------------------------------------------------

        print(
            "Connecting cameras..."
        )

        for name, camera in (
            raw_robot.cameras.items()
        ):
            print(
                f"  connecting camera: {name}"
            )

            camera.connect()

            connected_cameras.append(
                name
            )

        print(
            "Connected without "
            "SOFollower.configure()."
        )
        print()

        # =====================================================
        # 4. Read through OUR adapter
        # =====================================================

        observation = (
            robot.get_observation()
        )

        # =====================================================
        # 5. Robot state
        # =====================================================

        print(
            "========== ROBOT STATE =========="
        )

        q = observation.robot.q

        print(
            "q shape:",
            q.shape,
        )

        print(
            "q dtype:",
            q.dtype,
        )

        print(
            "q finite:",
            bool(
                np.all(
                    np.isfinite(q)
                )
            ),
        )

        print()

        for index, (
            key,
            value,
        ) in enumerate(
            zip(
                SO101_JOINT_KEYS,
                q,
                strict=True,
            )
        ):
            print(
                f"q[{index}] "
                f"{key:<20} "
                f"= {value:.6f}"
            )

        # =====================================================
        # 6. Camera frames
        # =====================================================

        print()
        print(
            "========== CAMERA FRAMES =========="
        )

        frames = (
            observation
            .cameras
            .frames
        )

        print(
            "camera keys:",
            tuple(
                frames.keys()
            ),
        )

        print()

        for key in SO101_CAMERA_KEYS:
            frame = frames[key]

            print(
                f"{key:<8} "
                f"shape={frame.shape}, "
                f"dtype={frame.dtype}, "
                f"min={frame.min()}, "
                f"max={frame.max()}, "
                f"mean={frame.mean():.2f}"
            )

        # =====================================================
        # 7. Timestamps
        # =====================================================

        print()
        print(
            "========== TIMESTAMPS =========="
        )

        robot_timestamp = (
            observation
            .robot
            .timestamp_s
        )

        camera_timestamp = (
            observation
            .cameras
            .timestamp_s
        )

        print(
            "robot timestamp:",
            robot_timestamp,
        )

        print(
            "camera timestamp:",
            camera_timestamp,
        )

        print(
            "same timestamp:",
            robot_timestamp
            == camera_timestamp,
        )

        # =====================================================
        # 8. Validation
        # =====================================================

        print()
        print(
            "========== VALIDATION =========="
        )

        # -----------------------------------------------------
        # Robot state validation
        # -----------------------------------------------------

        assert q.shape == (
            len(SO101_JOINT_KEYS),
        ), (
            "Unexpected SO-101 q shape: "
            f"{q.shape}"
        )

        assert np.all(
            np.isfinite(q)
        ), (
            "SO-101 q contains NaN or Inf"
        )

        # -----------------------------------------------------
        # Camera key validation
        # -----------------------------------------------------

        assert set(
            frames.keys()
        ) == set(
            SO101_CAMERA_KEYS
        ), (
            "Unexpected camera keys: "
            f"{tuple(frames.keys())}"
        )

        # -----------------------------------------------------
        # Camera data validation
        # -----------------------------------------------------

        for key in SO101_CAMERA_KEYS:
            frame = frames[key]

            camera_config = (
                hardware
                .cameras[key]
            )

            expected_shape = (
                camera_config.height,
                camera_config.width,
                3,
            )

            assert frame.shape == (
                expected_shape
            ), (
                f"Camera '{key}' "
                f"unexpected shape: "
                f"{frame.shape}, "
                f"expected "
                f"{expected_shape}"
            )

            assert frame.dtype == (
                np.uint8
            ), (
                f"Camera '{key}' "
                f"unexpected dtype: "
                f"{frame.dtype}"
            )

        # -----------------------------------------------------
        # Timestamp validation
        # -----------------------------------------------------

        assert (
            robot_timestamp
            == camera_timestamp
        ), (
            "RobotState and FrameBundle "
            "timestamps are different"
        )

        print(
            "All adapter checks passed."
        )

    finally:
        # =====================================================
        # 9. Cleanup
        #
        # 不调用：
        #
        #     robot.close()
        #
        # 因为这是 inspection-only 路径。
        #
        # 我们直接关闭 cameras 和 motor bus，
        # 并明确 disable_torque=False。
        # =====================================================

        print()
        print(
            "Disconnecting..."
        )

        # -----------------------------------------------------
        # Cameras
        #
        # 使用 reversed()，
        # 即使中途某台 camera.connect() 失败，
        # 前面已成功连接的 camera 也可以清理。
        # -----------------------------------------------------

        for name in reversed(
            connected_cameras
        ):
            try:
                raw_robot.cameras[
                    name
                ].disconnect()

                print(
                    f"  disconnected camera: "
                    f"{name}"
                )

            except Exception as exc:
                print(
                    f"  WARNING: failed to "
                    f"disconnect camera "
                    f"'{name}': {exc}"
                )

        # -----------------------------------------------------
        # Motor bus
        # -----------------------------------------------------

        if bus_connected:
            try:
                raw_robot.bus.disconnect(
                    disable_torque=False
                )

                print(
                    "  disconnected motor bus"
                )

            except Exception as exc:
                print(
                    "  WARNING: failed to "
                    "disconnect motor bus: "
                    f"{exc}"
                )

        print(
            "Disconnected."
        )

    print()
    print(
        "========================================"
    )
    print(
        "READ-ONLY ADAPTER TEST FINISHED"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()