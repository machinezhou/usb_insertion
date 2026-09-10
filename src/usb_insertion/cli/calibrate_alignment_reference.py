from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)


def project_root() -> Path:
    return Path(
        __file__
    ).resolve().parents[3]


def rgb_to_bgr(
    image: np.ndarray,
) -> np.ndarray:
    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )


def collect_two_points(
    image_bgr: np.ndarray,
    title: str,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
]:
    """
    第一点：
        USB 金属头最前端中心点。

    第二点：
        从 tip 指向 USB 塑料壳/线缆方向的点。
    """

    points: list[
        tuple[int, int]
    ] = []

    def callback(
        event,
        x,
        y,
        flags,
        param,
    ):
        del flags
        del param

        if (
            event
            == cv2.EVENT_LBUTTONDOWN
            and len(points) < 2
        ):
            points.append(
                (
                    int(x),
                    int(y),
                )
            )

    cv2.namedWindow(
        title,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        title,
        callback,
    )

    while len(points) < 2:
        display = (
            image_bgr.copy()
        )

        cv2.putText(
            display,
            (
                "Click 1: USB tip | "
                "Click 2: body direction | "
                "ESC: cancel"
            ),
            (
                10,
                30,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                255,
            ),
            2,
        )

        for point in points:
            cv2.circle(
                display,
                point,
                6,
                (
                    0,
                    0,
                    255,
                ),
                -1,
            )

        cv2.imshow(
            title,
            display,
        )

        key = (
            cv2.waitKey(20)
            & 0xFF
        )

        if key == 27:
            cv2.destroyWindow(
                title
            )

            raise KeyboardInterrupt(
                "Calibration cancelled"
            )

    cv2.destroyWindow(
        title
    )

    return (
        points[0],
        points[1],
    )


def angle_deg(
    p1,
    p2,
) -> float:
    return math.degrees(
        math.atan2(
            p2[1] - p1[1],
            p2[0] - p1[0],
        )
    )


def main() -> None:
    root = project_root()

    hardware = (
        load_hardware_config(
            root
            / "configs"
            / "hardware.yaml"
        )
    )

    robot = build_so101_adapter(
        hardware
    )

    raw_robot = (
        robot.raw_robot
    )

    output_dir = (
        root
        / "outputs"
        / "vision_alignment"
        / "calibration"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bus_connected = False

    connected_cameras: list[
        str
    ] = []

    try:
        print(
            "Connecting read-only "
            "inspection path..."
        )

        raw_robot.bus.connect()

        bus_connected = True

        for name, camera in (
            raw_robot.cameras.items()
        ):
            camera.connect()

            connected_cameras.append(
                name
            )

        # 额外读取几帧稳定相机
        for _ in range(5):
            observation = (
                robot.get_observation()
            )

        result = {}

        for camera_name in (
            "top",
            "side",
        ):
            frame_rgb = (
                observation
                .cameras
                .frames[
                    camera_name
                ]
            )

            frame_bgr = (
                rgb_to_bgr(
                    frame_rgb
                )
            )

            title = (
                f"{camera_name}: "
                "select USB template ROI"
            )

            roi_raw = (
                cv2.selectROI(
                    title,
                    frame_bgr,
                    showCrosshair=True,
                    fromCenter=False,
                )
            )

            cv2.destroyWindow(
                title
            )

            x, y, width, height = (
                int(value)
                for value
                in roi_raw
            )

            if (
                width <= 0
                or height <= 0
            ):
                raise RuntimeError(
                    "Invalid ROI selected"
                )

            template_bgr = (
                frame_bgr[
                    y:y + height,
                    x:x + width,
                ]
                .copy()
            )

            tip_local, axis_local = (
                collect_two_points(
                    template_bgr,
                    (
                        f"{camera_name}: "
                        "USB points"
                    ),
                )
            )

            target_tip = (
                x + tip_local[0],
                y + tip_local[1],
            )

            target_axis = angle_deg(
                tip_local,
                axis_local,
            )

            template_path = (
                output_dir
                / (
                    f"{camera_name}"
                    "_plug_template.png"
                )
            )

            cv2.imwrite(
                str(
                    template_path
                ),
                template_bgr,
            )

            reference = (
                frame_bgr.copy()
            )

            cv2.rectangle(
                reference,
                (
                    x,
                    y,
                ),
                (
                    x + width,
                    y + height,
                ),
                (
                    0,
                    255,
                    255,
                ),
                2,
            )

            cv2.drawMarker(
                reference,
                target_tip,
                (
                    0,
                    255,
                    0,
                ),
                cv2.MARKER_CROSS,
                24,
                2,
            )

            reference_path = (
                output_dir
                / (
                    f"{camera_name}"
                    "_reference.png"
                )
            )

            cv2.imwrite(
                str(
                    reference_path
                ),
                reference,
            )

            relative_template = (
                template_path
                .relative_to(root)
            )

            result[
                camera_name
            ] = {
                "roi": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                },

                "template_path": str(
                    relative_template
                ),

                "template_tip_xy": [
                    tip_local[0],
                    tip_local[1],
                ],

                "template_axis_xy": [
                    axis_local[0],
                    axis_local[1],
                ],

                "target_tip_xy": [
                    target_tip[0],
                    target_tip[1],
                ],

                "target_axis_angle_deg": float(
                    target_axis
                ),

                # 以下是 V1 初始值，
                # 后续根据真实标定调整。
                "min_matches": 8,
                "ratio_test": 0.75,
                "ransac_reproj_threshold": 3.0,

                "capture_tip_distance_px": 120.0,
                "capture_angle_deg": 20.0,

                "aligned_dx_px": 6.0,
                "aligned_dy_px": 6.0,
                "aligned_angle_deg": 3.0,
            }

        config = {
            "correct_face_label": (
                "marked"
            ),

            "cameras": result,

            "servo": {
                "error_keys": [
                    "top_dx_px",
                    "top_dy_px",
                    "top_angle_deg",
                    "side_dx_px",
                    "side_dy_px",
                    "side_angle_deg",
                ],

                "controlled_joint_indices": [
                    0,
                    1,
                    2,
                    3,
                    4,
                ],

                "jacobian_path": (
                    "outputs/"
                    "vision_alignment/"
                    "calibration/"
                    "image_jacobian.npy"
                ),

                # 初始保守值
                "gain": 0.35,
                "damping": 0.1,

                # SO-101 use_degrees=True
                # 每个视觉控制周期最多变化 0.4°
                "max_joint_delta": [
                    0.4,
                    0.4,
                    0.4,
                    0.4,
                    0.4,
                ],
            },
        }

        config_path = (
            root
            / "configs"
            / "vision.yaml"
        )

        config_path.write_text(
            yaml.safe_dump(
                config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "Calibration saved:"
        )
        print(
            config_path
        )

        print()
        print(
            "Templates:"
        )

        print(
            output_dir
        )

    finally:
        for name in reversed(
            connected_cameras
        ):
            try:
                raw_robot.cameras[
                    name
                ].disconnect()
            except Exception:
                pass

        if bus_connected:
            try:
                raw_robot.bus.disconnect(
                    disable_torque=False
                )
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()