from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sanitize_label(
    label: str,
) -> str:
    """
    将用户 label 转换成安全的目录名。
    """

    cleaned = "".join(
        char
        if char.isalnum()
        or char in ("-", "_")
        else "_"
        for char in label
    )

    cleaned = cleaned.strip("_")

    return cleaned or "unlabeled"


def save_rgb_png(
    path: Path,
    frame: np.ndarray,
) -> None:
    """
    LeRobot 当前 OpenCV camera 输出为 RGB ndarray。

    cv2.imwrite() 按 BGR 解释，
    因此保存 PNG 前先 RGB -> BGR。
    """

    if frame.ndim != 3:
        raise ValueError(
            "Expected HxWxC color frame"
        )

    if frame.shape[2] != 3:
        raise ValueError(
            "Expected 3-channel frame"
        )

    bgr = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR,
    )

    success = cv2.imwrite(
        str(path),
        bgr,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one real SO-101 "
            "multi-camera vision snapshot."
        )
    )

    parser.add_argument(
        "--label",
        default="unlabeled",
        help=(
            "Scene label, for example: "
            "unplugged, partial, inserted"
        ),
    )

    args = parser.parse_args()

    label = sanitize_label(
        args.label
    )

    project_root = (
        get_project_root()
    )

    hardware_path = (
        project_root
        / "configs"
        / "hardware.yaml"
    )

    hardware = load_hardware_config(
        hardware_path
    )

    robot = build_so101_adapter(
        hardware
    )

    raw_robot = robot.raw_robot

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        project_root
        / "outputs"
        / "vision_snapshots"
        / label
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "========================================"
    )
    print(
        "USB INSERTION - VISION SNAPSHOT"
    )
    print(
        "========================================"
    )

    print()
    print(
        f"label: {label}"
    )

    print(
        f"output: {output_dir}"
    )

    print()
    print(
        "This tool does NOT call send_action()."
    )

    print(
        "SOFollower.configure() is bypassed."
    )

    print()

    bus_connected = False
    connected_cameras: list[str] = []

    try:
        # -----------------------------------------------------
        # Inspection-only connection.
        # -----------------------------------------------------

        print(
            "Connecting motor bus..."
        )

        raw_robot.bus.connect()

        bus_connected = True

        print(
            "Connecting cameras..."
        )

        for name, camera in (
            raw_robot.cameras.items()
        ):
            print(
                f"  {name}"
            )

            camera.connect()

            connected_cameras.append(
                name
            )

        print()
        print(
            "Capturing observation..."
        )

        # 第一帧丢弃，作为额外稳定帧。
        robot.get_observation()

        observation = (
            robot.get_observation()
        )

        q = observation.robot.q

        frames = (
            observation
            .cameras
            .frames
        )

        # -----------------------------------------------------
        # Save camera data.
        #
        # PNG:
        #   方便肉眼查看。
        #
        # NPY:
        #   保存 Adapter 返回的原始 RGB ndarray，
        #   后续离线算法测试可以精确复现。
        # -----------------------------------------------------

        camera_metadata = {}

        for name, frame in (
            frames.items()
        ):
            png_path = (
                output_dir
                / f"{name}.png"
            )

            npy_path = (
                output_dir
                / f"{name}.npy"
            )

            save_rgb_png(
                png_path,
                frame,
            )

            np.save(
                npy_path,
                frame,
                allow_pickle=False,
            )

            camera_metadata[name] = {
                "shape": list(
                    frame.shape
                ),
                "dtype": str(
                    frame.dtype
                ),
                "min": int(
                    frame.min()
                ),
                "max": int(
                    frame.max()
                ),
                "mean": float(
                    frame.mean()
                ),
            }

            print(
                f"saved {name}: "
                f"{frame.shape}"
            )

        # -----------------------------------------------------
        # Save metadata.
        # -----------------------------------------------------

        metadata = {
            "label": label,

            "hardware_config": str(
                hardware_path
            ),

            "robot_q": [
                float(value)
                for value in q
            ],

            "robot_timestamp_s": float(
                observation
                .robot
                .timestamp_s
            ),

            "camera_timestamp_s": float(
                observation
                .cameras
                .timestamp_s
            ),

            "cameras": camera_metadata,
        }

        metadata_path = (
            output_dir
            / "metadata.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "Snapshot complete."
        )

        print(
            f"Metadata: {metadata_path}"
        )

    finally:
        print()
        print(
            "Disconnecting..."
        )

        for name in reversed(
            connected_cameras
        ):
            try:
                raw_robot.cameras[
                    name
                ].disconnect()

            except Exception as exc:
                print(
                    f"WARNING: camera "
                    f"{name} disconnect failed: "
                    f"{exc}"
                )

        if bus_connected:
            try:
                raw_robot.bus.disconnect(
                    disable_torque=False
                )

            except Exception as exc:
                print(
                    "WARNING: motor bus "
                    "disconnect failed: "
                    f"{exc}"
                )

        print(
            "Disconnected."
        )

    print()
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()