from __future__ import annotations

import argparse
from pathlib import Path

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)
from usb_insertion.vision.insertion.path import (
    InsertionPose,
    save_insertion_pose,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture SO-101 joint state "
            "for an insertion calibration pose."
        )
    )

    parser.add_argument(
        "--stage",
        choices=(
            "preinsert",
            "inserted",
        ),
        required=True,
        help=(
            "Which insertion calibration "
            "pose is being captured."
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

    robot = build_so101_adapter(
        hardware
    )

    raw_robot = robot.raw_robot

    bus_connected = False

    connected_cameras: list[
        str
    ] = []

    try:
        # -----------------------------------------------------
        # Read-only inspection path.
        #
        # This utility never sends an action.
        # -----------------------------------------------------

        print(
            "========================================"
        )

        print(
            "USB INSERTION POSE CAPTURE"
        )

        print(
            "========================================"
        )

        print()

        print(
            f"stage: {args.stage}"
        )

        print()

        print(
            "This program DOES NOT "
            "send robot actions."
        )

        print(
            "Move/place the arm manually "
            "into the requested state "
            "before capturing."
        )

        print()

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

        print()
        print(
            "Requested stage:"
        )

        print(
            f"    {args.stage}"
        )

        if (
            args.stage
            == "preinsert"
        ):
            print()
            print(
                "Place the USB in the "
                "ideal PREINSERT pose:"
            )

            print(
                "- correctly aligned"
            )

            print(
                "- correct plug face"
            )

            print(
                "- still outside the port"
            )

        else:
            print()
            print(
                "Place the USB in the "
                "fully INSERTED pose."
            )

        print()
        print(
            "No command will be sent."
        )

        input(
            "Press Enter to capture..."
        )

        observation = (
            robot.get_observation()
        )

        output_path = (
            root
            / "outputs"
            / "insertion"
            / (
                f"{args.stage}"
                "_pose.json"
            )
        )

        pose = InsertionPose(
            stage=args.stage,
            q=(
                observation
                .robot
                .q
                .copy()
            ),
            timestamp_s=(
                observation
                .robot
                .timestamp_s
            ),
        )

        save_insertion_pose(
            output_path,
            pose,
        )

        print()
        print(
            "Captured q:"
        )

        for index, value in enumerate(
            pose.q
        ):
            print(
                f"  q[{index}] = "
                f"{value:.6f}"
            )

        print()
        print(
            "Saved:"
        )

        print(
            output_path
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

        print(
            "Disconnected."
        )


if __name__ == "__main__":
    main()
