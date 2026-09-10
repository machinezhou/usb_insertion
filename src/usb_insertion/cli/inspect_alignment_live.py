from __future__ import annotations

from pathlib import Path

import cv2

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)
from usb_insertion.vision.alignment.config import (
    load_vision_alignment_config,
)
from usb_insertion.vision.alignment.estimator import (
    TemplateAlignmentEstimator,
)


def project_root() -> Path:
    return Path(
        __file__
    ).resolve().parents[3]


def main() -> None:
    root = project_root()

    hardware = (
        load_hardware_config(
            root
            / "configs"
            / "hardware.yaml"
        )
    )

    vision_config = (
        load_vision_alignment_config(
            root
            / "configs"
            / "vision.yaml"
        )
    )

    estimator = (
        TemplateAlignmentEstimator(
            vision_config
        )
    )

    robot = build_so101_adapter(
        hardware
    )

    raw_robot = (
        robot.raw_robot
    )

    bus_connected = False

    connected_cameras: list[
        str
    ] = []

    try:
        raw_robot.bus.connect()

        bus_connected = True

        for name, camera in (
            raw_robot.cameras.items()
        ):
            camera.connect()

            connected_cameras.append(
                name
            )

        print()
        print(
            "Alignment live inspection"
        )

        print(
            "Press q to quit."
        )

        print()

        while True:
            observation = (
                robot.get_observation()
            )

            alignment = (
                estimator.estimate(
                    observation
                )
            )

            errors = " | ".join(
                (
                    f"{key}="
                    f"{value:.2f}"
                )
                for key, value
                in alignment.errors.items()
            )

            print(
                "\r"
                f"visible={alignment.visible} "
                f"capture="
                f"{alignment.in_capture_region} "
                f"aligned="
                f"{alignment.aligned} "
                f"confidence="
                f"{alignment.confidence:.2f} "
                f"{errors}",
                end="",
                flush=True,
            )

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

                overlay_rgb = (
                    estimator.draw_debug(
                        frame_rgb,
                        camera_name,
                    )
                )

                overlay_bgr = (
                    cv2.cvtColor(
                        overlay_rgb,
                        cv2.COLOR_RGB2BGR,
                    )
                )

                cv2.imshow(
                    (
                        "USB Alignment - "
                        f"{camera_name}"
                    ),
                    overlay_bgr,
                )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

    finally:
        print()

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