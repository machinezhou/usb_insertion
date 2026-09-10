from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--required-consecutive",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--tolerance-scale",
        type=float,
        default=4.0,
    )

    args = parser.parse_args()

    root = project_root()

    hardware = (
        load_hardware_config(
            root
            / "configs"
            / "hardware.yaml"
        )
    )

    vision = (
        load_vision_alignment_config(
            root
            / "configs"
            / "vision.yaml"
        )
    )

    estimator = (
        TemplateAlignmentEstimator(
            vision
        )
    )

    robot = build_so101_adapter(
        hardware
    )

    raw_robot = (
        robot.raw_robot
    )

    bus_connected = False
    connected_cameras = []

    samples = []

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

        print(
            "Place USB in a NORMAL "
            "fully inserted state."
        )

        input(
            "Press ENTER to capture "
            "success reference..."
        )

        for index in range(
            args.samples
        ):
            observation = (
                robot.get_observation()
            )

            alignment = (
                estimator.estimate(
                    observation
                )
            )

            if not alignment.visible:
                raise RuntimeError(
                    "USB tracking lost "
                    f"at sample {index}"
                )

            samples.append(
                dict(
                    alignment.errors
                )
            )

            time.sleep(
                0.05
            )

        keys = tuple(
            samples[0].keys()
        )

        matrix = np.asarray(
            [
                [
                    sample[key]
                    for key in keys
                ]
                for sample in samples
            ],
            dtype=np.float64,
        )

        mean = (
            matrix.mean(
                axis=0
            )
        )

        std = (
            matrix.std(
                axis=0
            )
        )

        target_errors = {
            key: float(value)
            for key, value
            in zip(
                keys,
                mean,
                strict=True,
            )
        }

        # 只是根据静态重复波动产生初始容差。
        # 最终值下一阶段还要用
        # almost-inserted 负样本重新确定。
        tolerances = {
            key: float(
                max(
                    args.tolerance_scale
                    * sigma,
                    1.0,
                )
            )
            for key, sigma
            in zip(
                keys,
                std,
                strict=True,
            )
        }

        output = {
            "required_consecutive": (
                args
                .required_consecutive
            ),
            "target_errors": (
                target_errors
            ),
            "tolerances": (
                tolerances
            ),
        }

        output_path = (
            root
            / "configs"
            / "success_reference.yaml"
        )

        output_path.write_text(
            yaml.safe_dump(
                output,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        print(
            "Saved:",
            output_path,
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


if __name__ == "__main__":
    main()
