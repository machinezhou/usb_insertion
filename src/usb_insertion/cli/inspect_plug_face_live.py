from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)
from usb_insertion.vision.alignment.face import (
    TemplatePlugFaceClassifier,
    load_plug_face_config,
)


def project_root() -> Path:
    return Path(
        __file__
    ).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live inspection tool for "
            "USB plug-face classification."
        )
    )

    parser.add_argument(
        "--hardware",
        default=(
            "configs/hardware.yaml"
        ),
        help=(
            "Hardware config path "
            "relative to project root."
        ),
    )

    parser.add_argument(
        "--config",
        default=(
            "configs/plug_face.yaml"
        ),
        help=(
            "Plug-face config path "
            "relative to project root."
        ),
    )

    args = parser.parse_args()

    root = project_root()

    hardware_path = Path(
        args.hardware
    )

    if not hardware_path.is_absolute():
        hardware_path = (
            root
            / hardware_path
        )

    config_path = Path(
        args.config
    )

    if not config_path.is_absolute():
        config_path = (
            root
            / config_path
        )

    hardware = (
        load_hardware_config(
            hardware_path
        )
    )

    face_config = (
        load_plug_face_config(
            config_path
        )
    )

    if (
        face_config.camera_name
        not in hardware.cameras
    ):
        raise ValueError(
            "Plug-face camera "
            f"'{face_config.camera_name}' "
            "is not present in "
            "hardware configuration"
        )

    classifier = (
        TemplatePlugFaceClassifier(
            face_config
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

    camera_name = (
        face_config.camera_name
    )

    roi = face_config.roi

    print(
        "========================================"
    )
    print(
        "USB INSERTION - PLUG FACE LIVE INSPECT"
    )
    print(
        "========================================"
    )
    print()

    print(
        f"hardware: {hardware_path}"
    )

    print(
        f"config:   {config_path}"
    )

    print(
        f"camera:   {camera_name}"
    )

    print(
        "feature:  "
        f"{classifier.feature_type}"
    )

    print(
        "correct:  "
        f"{face_config.correct_label}"
    )

    print()

    print(
        "This tool is inspection-only."
    )

    print(
        "It does NOT call send_action()."
    )

    print(
        "SOFollower.configure() is bypassed."
    )

    print()

    try:
        # -----------------------------------------------------
        # Read-only hardware connection.
        #
        # We intentionally do not call robot.connect(),
        # because LeRobot robot.connect() may perform
        # configuration operations.
        #
        # The motor bus is connected only because
        # robot.get_observation() also reads joint state.
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
            camera.connect()

            connected_cameras.append(
                name
            )

            print(
                f"  connected: {name}"
            )

        print()
        print(
            "Warming up cameras..."
        )

        for _ in range(5):
            robot.get_observation()

        print()
        print(
            "Live classification started."
        )

        print(
            "Show MARKED and UNMARKED "
            "faces alternately."
        )

        print(
            "Press q or Esc to quit."
        )

        print()

        while True:
            observation = (
                robot.get_observation()
            )

            frame_rgb = (
                observation
                .cameras
                .frames[
                    camera_name
                ]
            )

            result = (
                classifier.classify(
                    frame_rgb
                )
            )

            display_bgr = (
                cv2.cvtColor(
                    frame_rgb,
                    cv2.COLOR_RGB2BGR,
                )
            )

            # -------------------------------------------------
            # Draw configured classification ROI.
            # -------------------------------------------------

            cv2.rectangle(
                display_bgr,
                (
                    roi.x,
                    roi.y,
                ),
                (
                    roi.x2,
                    roi.y2,
                ),
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

            label_text = (
                result.label
                if result.label is not None
                else "unknown"
            )

            correct = (
                result.label
                == face_config.correct_label
            )

            if result.label is None:
                face_status = (
                    "UNKNOWN"
                )
            elif correct:
                face_status = (
                    "CORRECT"
                )
            else:
                face_status = (
                    "WRONG"
                )

            score_text = " ".join(
                (
                    f"{label}="
                    f"{score:.3f}"
                )
                for label, score
                in sorted(
                    result
                    .scores
                    .items()
                )
            )

            terminal_line = (
                f"visible={result.visible} "
                f"label={label_text} "
                f"status={face_status} "
                f"confidence="
                f"{result.confidence:.3f} "
                f"reason={result.reason}"
            )

            if score_text:
                terminal_line += (
                    " | "
                    + score_text
                )

            print(
                "\r"
                + terminal_line
                + " " * 20,
                end="",
                flush=True,
            )

            # -------------------------------------------------
            # On-screen status.
            # -------------------------------------------------

            lines = [
                (
                    "label: "
                    f"{label_text}"
                ),
                (
                    "status: "
                    f"{face_status}"
                ),
                (
                    "confidence: "
                    f"{result.confidence:.3f}"
                ),
                (
                    "reason: "
                    f"{result.reason}"
                ),
            ]

            if score_text:
                lines.append(
                    score_text
                )

            y = 30

            for text in lines:
                cv2.putText(
                    display_bgr,
                    text,
                    (
                        10,
                        y,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (
                        255,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

                y += 28

            cv2.imshow(
                (
                    "USB Plug Face - "
                    f"{camera_name}"
                ),
                display_bgr,
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
                    "  disconnected camera: "
                    f"{name}"
                )

            except Exception as exc:
                print(
                    "WARNING: failed to "
                    "disconnect camera "
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
