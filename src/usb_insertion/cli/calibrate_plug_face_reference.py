from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_bgr(
    robot,
    camera_name: str,
):
    observation = robot.get_observation()

    frame_rgb = (
        observation
        .cameras
        .frames[
            camera_name
        ]
    )

    return cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2BGR,
    )


def capture_template(
    robot,
    camera_name: str,
    roi: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    x, y, width, height = roi

    frame = read_bgr(
        robot,
        camera_name,
    )

    crop = (
        frame[
            y:y + height,
            x:x + width,
        ]
        .copy()
    )

    if crop.size == 0:
        raise RuntimeError(
            "Selected plug-face ROI is empty"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        crop,
    )

    if not success:
        raise RuntimeError(
            "Failed to save template: "
            f"{output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture marked/unmarked USB "
            "plug-face reference templates."
        )
    )

    parser.add_argument(
        "--camera",
        default="wrist",
        help=(
            "Camera used for USB face "
            "classification."
        ),
    )

    args = parser.parse_args()

    root = project_root()

    hardware_path = (
        root
        / "configs"
        / "hardware.yaml"
    )

    hardware = load_hardware_config(
        hardware_path
    )

    if (
        args.camera
        not in hardware.cameras
    ):
        raise ValueError(
            "Unknown camera: "
            f"{args.camera}"
        )

    robot = build_so101_adapter(
        hardware
    )

    raw_robot = robot.raw_robot

    output_dir = (
        root
        / "outputs"
        / "vision_alignment"
        / "plug_face"
    )

    marked_path = (
        output_dir
        / "marked.png"
    )

    unmarked_path = (
        output_dir
        / "unmarked.png"
    )

    bus_connected = False

    connected_cameras: list[
        str
    ] = []

    try:
        # -----------------------------------------------------
        # Read-only hardware path.
        #
        # Do NOT call SOFollower.connect(),
        # because this utility never sends actions.
        # -----------------------------------------------------

        print(
            "Connecting motor bus "
            "without robot.configure()..."
        )

        raw_robot.bus.connect()

        bus_connected = True

        print(
            "Connecting cameras..."
        )

        for name, camera in (
            raw_robot.cameras.items()
        ):
            camera.connect()

            connected_cameras.append(
                name
            )

            print(
                f"  connected: {name}"
            )

        # Camera warm-up.
        for _ in range(5):
            robot.get_observation()

        # -----------------------------------------------------
        # Marked face.
        # -----------------------------------------------------

        print()
        print(
            "Show the MARKED / correct "
            "USB face to the camera."
        )

        print(
            "Keep the USB inside the "
            "future classification region."
        )

        input(
            "Press Enter when ready..."
        )

        frame = read_bgr(
            robot,
            args.camera,
        )

        roi_raw = cv2.selectROI(
            "Select USB plug-face ROI",
            frame,
            showCrosshair=True,
            fromCenter=False,
        )

        cv2.destroyWindow(
            "Select USB plug-face ROI"
        )

        x, y, width, height = (
            int(value)
            for value in roi_raw
        )

        if (
            width <= 0
            or height <= 0
        ):
            raise RuntimeError(
                "Invalid ROI selected"
            )

        roi = (
            x,
            y,
            width,
            height,
        )

        capture_template(
            robot=robot,
            camera_name=args.camera,
            roi=roi,
            output_path=marked_path,
        )

        print(
            "Saved marked template:"
        )

        print(
            marked_path
        )

        # -----------------------------------------------------
        # Unmarked face.
        # -----------------------------------------------------

        print()
        print(
            "Now show the UNMARKED / "
            "opposite USB face."
        )

        print(
            "Keep the USB in the same ROI."
        )

        input(
            "Press Enter when ready..."
        )

        capture_template(
            robot=robot,
            camera_name=args.camera,
            roi=roi,
            output_path=unmarked_path,
        )

        print(
            "Saved unmarked template:"
        )

        print(
            unmarked_path
        )

        # -----------------------------------------------------
        # Build plug_face.yaml
        # -----------------------------------------------------

        config = {
            "camera_name": (
                args.camera
            ),

            "correct_label": (
                "marked"
            ),

            "roi": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            },

            "templates": {
                "marked": str(
                    marked_path
                    .relative_to(root)
                ),

                "unmarked": str(
                    unmarked_path
                    .relative_to(root)
                ),
            },

            # Initial values only.
            # These will be validated later
            # during the calibration phase.
            "min_good_matches": 6,
            "ratio_test": 0.75,
            "min_score_margin": 0.15,
        }

        config_path = (
            root
            / "configs"
            / "plug_face.yaml"
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
            "Plug-face calibration saved:"
        )

        print(
            config_path
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

                print(
                    f"  disconnected camera: "
                    f"{name}"
                )

            except Exception as exc:
                print(
                    "WARNING: failed to "
                    f"disconnect camera "
                    f"'{name}': {exc}"
                )

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
                    "WARNING: failed to "
                    "disconnect motor bus: "
                    f"{exc}"
                )

        cv2.destroyAllWindows()

        print(
            "Disconnected."
        )


if __name__ == "__main__":
    main()
