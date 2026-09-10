from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)
from usb_insertion.control.action_guard import (
    ActionGuard,
    ActionGuardConfig,
)
from usb_insertion.control.scripted_motion import (
    ScriptedMotionController,
)
from usb_insertion.control.trajectory_planner import (
    TrajectoryLimits,
)
from usb_insertion.vision.alignment.config import (
    load_vision_alignment_config,
)
from usb_insertion.vision.alignment.estimator import (
    TemplateAlignmentEstimator,
)


def project_root() -> Path:
    """
    返回项目根目录：

        ~/projects/usb_insertion
    """
    return Path(__file__).resolve().parents[3]


def alignment_error_vector(
    alignment,
    error_keys: tuple[str, ...],
) -> np.ndarray:
    """
    将 AlignmentEstimate.errors 按固定顺序
    转换为视觉误差向量。

    例如：

        [
            top_dx_px,
            top_dy_px,
            top_angle_deg,
            side_dx_px,
            side_dy_px,
            side_angle_deg,
        ]
    """

    missing = [
        key
        for key in error_keys
        if key not in alignment.errors
    ]

    if missing:
        raise KeyError(
            "AlignmentEstimate missing "
            f"error keys: {missing}"
        )

    vector = np.asarray(
        [
            float(
                alignment.errors[key]
            )
            for key in error_keys
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(vector)
    ):
        raise ValueError(
            "Alignment error contains "
            "NaN or Inf"
        )

    return vector


def measure_mean_error(
    *,
    robot,
    estimator: TemplateAlignmentEstimator,
    error_keys: tuple[str, ...],
    samples: int,
    sample_interval_s: float,
) -> np.ndarray:
    """
    连续读取多帧 Vision Alignment，
    返回平均视觉误差。

    Jacobian 标定使用多帧平均，
    降低单帧检测抖动对结果的影响。
    """

    if samples <= 0:
        raise ValueError(
            "samples must be > 0"
        )

    measurements: list[
        np.ndarray
    ] = []

    for _ in range(samples):
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
                "USB visual target lost "
                "during Jacobian calibration: "
                f"{alignment.reason}"
            )

        measurements.append(
            alignment_error_vector(
                alignment,
                error_keys,
            )
        )

        if sample_interval_s > 0:
            time.sleep(
                sample_interval_s
            )

    return np.mean(
        np.stack(
            measurements,
            axis=0,
        ),
        axis=0,
    )


def print_error_vector(
    title: str,
    error_keys: tuple[str, ...],
    error: np.ndarray,
) -> None:
    print()
    print(title)

    for key, value in zip(
        error_keys,
        error,
        strict=True,
    ):
        print(
            f"  {key:<24} "
            f"{value:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the local visual-servo "
            "image Jacobian for SO-101."
        )
    )

    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help=(
            "Required safety acknowledgement. "
            "Without this flag the robot will "
            "not move."
        ),
    )

    parser.add_argument(
        "--delta-deg",
        type=float,
        default=0.8,
        help=(
            "Central-difference joint "
            "perturbation amplitude in degrees."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help=(
            "Number of visual measurements "
            "averaged at each perturbation."
        ),
    )

    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.25,
        help=(
            "Wait after each robot movement "
            "before measuring vision."
        ),
    )

    parser.add_argument(
        "--move-duration-s",
        type=float,
        default=1.0,
        help=(
            "Requested duration for a "
            "single small perturbation move."
        ),
    )

    parser.add_argument(
        "--return-duration-s",
        type=float,
        default=1.5,
        help=(
            "Requested duration for returning "
            "to baseline."
        ),
    )

    parser.add_argument(
        "--max-step-deg",
        type=float,
        default=0.25,
        help=(
            "Maximum commanded joint delta "
            "per control cycle."
        ),
    )

    parser.add_argument(
        "--max-velocity-deg-s",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-acceleration-deg-s2",
        type=float,
        default=20.0,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate CLI configuration before touching hardware.
    # ---------------------------------------------------------

    if not args.enable_motion:
        raise RuntimeError(
            "\n"
            "This command intentionally MOVES "
            "the SO-101.\n"
            "\n"
            "Do not run Jacobian calibration yet "
            "unless Vision Alignment has already "
            "been verified.\n"
            "\n"
            "When ready, explicitly add:\n"
            "\n"
            "    --enable-motion\n"
        )

    if args.delta_deg <= 0:
        raise ValueError(
            "--delta-deg must be > 0"
        )

    if args.samples <= 0:
        raise ValueError(
            "--samples must be > 0"
        )

    if args.sample_interval_s < 0:
        raise ValueError(
            "--sample-interval-s must be >= 0"
        )

    if args.settle_s < 0:
        raise ValueError(
            "--settle-s must be >= 0"
        )

    if args.move_duration_s <= 0:
        raise ValueError(
            "--move-duration-s must be > 0"
        )

    if args.return_duration_s <= 0:
        raise ValueError(
            "--return-duration-s must be > 0"
        )

    if args.max_step_deg <= 0:
        raise ValueError(
            "--max-step-deg must be > 0"
        )

    if args.max_velocity_deg_s <= 0:
        raise ValueError(
            "--max-velocity-deg-s must be > 0"
        )

    if args.max_acceleration_deg_s2 <= 0:
        raise ValueError(
            "--max-acceleration-deg-s2 "
            "must be > 0"
        )

    # ---------------------------------------------------------
    # Project configuration.
    # ---------------------------------------------------------

    root = project_root()

    hardware_path = (
        root
        / "configs"
        / "hardware.yaml"
    )

    vision_path = (
        root
        / "configs"
        / "vision.yaml"
    )

    hardware = (
        load_hardware_config(
            hardware_path
        )
    )

    vision = (
        load_vision_alignment_config(
            vision_path
        )
    )

    error_keys = (
        vision.servo.error_keys
    )

    controlled_joints = (
        vision
        .servo
        .controlled_joint_indices
    )

    if not controlled_joints:
        raise ValueError(
            "No controlled joints configured"
        )

    if len(
        set(controlled_joints)
    ) != len(controlled_joints):
        raise ValueError(
            "controlled_joint_indices "
            "contains duplicates"
        )

    if 5 in controlled_joints:
        raise ValueError(
            "gripper joint index 5 must not "
            "be part of image Jacobian "
            "calibration"
        )

    # ---------------------------------------------------------
    # Build Vision and Robot.
    # ---------------------------------------------------------

    estimator = (
        TemplateAlignmentEstimator(
            vision
        )
    )

    robot = build_so101_adapter(
        hardware
    )

    connected = False

    motion: (
        ScriptedMotionController
        | None
    ) = None

    baseline_q: (
        np.ndarray
        | None
    ) = None

    calibration_completed = False

    print()
    print(
        "========================================"
    )
    print(
        "SO-101 IMAGE JACOBIAN CALIBRATION"
    )
    print(
        "========================================"
    )

    print()
    print(
        "WARNING:"
    )

    print(
        "This command WILL move the robot."
    )

    print(
        "The USB must already be in the "
        "verified visual pre-insert pose."
    )

    print()
    print(
        f"delta_deg               : "
        f"{args.delta_deg}"
    )

    print(
        f"samples                 : "
        f"{args.samples}"
    )

    print(
        f"max_step_deg            : "
        f"{args.max_step_deg}"
    )

    print(
        f"max_velocity_deg_s      : "
        f"{args.max_velocity_deg_s}"
    )

    print(
        f"max_acceleration_deg_s2 : "
        f"{args.max_acceleration_deg_s2}"
    )

    print()

    try:
        # -----------------------------------------------------
        # Unlike inspection tools, this is a REAL motion tool.
        #
        # Therefore use normal LeRobot connect().
        #
        # hardware.yaml already specifies:
        #
        #     calibrate_on_connect: false
        #
        # so existing calibration is used.
        # -----------------------------------------------------

        print(
            "Connecting SO-101 using "
            "normal LeRobot motion path..."
        )

        robot.connect()

        connected = True

        print(
            "Connected."
        )

        # -----------------------------------------------------
        # Verify initial visual condition.
        # -----------------------------------------------------

        print()
        print(
            "Checking baseline visual alignment..."
        )

        baseline_observation = (
            robot.get_observation()
        )

        baseline_alignment = (
            estimator.estimate(
                baseline_observation
            )
        )

        print(
            f"visible           : "
            f"{baseline_alignment.visible}"
        )

        print(
            f"in_capture_region : "
            f"{baseline_alignment.in_capture_region}"
        )

        print(
            f"aligned           : "
            f"{baseline_alignment.aligned}"
        )

        print(
            f"confidence        : "
            f"{baseline_alignment.confidence:.3f}"
        )

        print(
            f"reason            : "
            f"{baseline_alignment.reason}"
        )

        if not (
            baseline_alignment.visible
        ):
            raise RuntimeError(
                "Calibration refused: "
                "USB is not reliably visible."
            )

        if not (
            baseline_alignment
            .in_capture_region
        ):
            raise RuntimeError(
                "Calibration refused: "
                "USB is outside Visual Servo "
                "capture region."
            )

        if not (
            baseline_alignment.aligned
        ):
            raise RuntimeError(
                "Calibration refused: "
                "USB is not in the calibrated "
                "pre-insert alignment pose."
            )

        baseline_q = (
            baseline_observation
            .robot
            .q
            .copy()
        )

        if baseline_q.shape != (
            6,
        ):
            raise ValueError(
                "Expected SO-101 q shape (6,), "
                f"got {baseline_q.shape}"
            )

        if not np.all(
            np.isfinite(
                baseline_q
            )
        ):
            raise ValueError(
                "Baseline robot q contains "
                "NaN or Inf"
            )

        print()
        print(
            "Baseline q:"
        )

        for index, value in enumerate(
            baseline_q
        ):
            print(
                f"  q[{index}] = "
                f"{value:.6f}"
            )

        # -----------------------------------------------------
        # Keep current gripper position fixed throughout
        # calibration.
        # -----------------------------------------------------

        fixed_gripper_position = float(
            baseline_q[5]
        )

        max_delta = np.full(
            6,
            args.max_step_deg,
            dtype=np.float64,
        )

        guard = ActionGuard(
            ActionGuardConfig(
                gripper_index=5,
                fixed_gripper_position=(
                    fixed_gripper_position
                ),
                max_delta_per_step=(
                    max_delta
                ),
            )
        )

        motion = (
            ScriptedMotionController(
                robot=robot,
                action_guard=guard,
                control_fps=30.0,
                trajectory_limits=(
                    TrajectoryLimits(
                        max_velocity=np.full(
                            6,
                            args
                            .max_velocity_deg_s,
                            dtype=np.float64,
                        ),
                        max_acceleration=np.full(
                            6,
                            args
                            .max_acceleration_deg_s2,
                            dtype=np.float64,
                        ),
                        max_delta_per_step=(
                            max_delta
                        ),
                    )
                ),
            )
        )

        # -----------------------------------------------------
        # Baseline visual measurement.
        # -----------------------------------------------------

        baseline_error = (
            measure_mean_error(
                robot=robot,
                estimator=estimator,
                error_keys=error_keys,
                samples=args.samples,
                sample_interval_s=(
                    args.sample_interval_s
                ),
            )
        )

        print_error_vector(
            "Baseline visual error:",
            error_keys,
            baseline_error,
        )

        # -----------------------------------------------------
        # Central-difference Image Jacobian.
        #
        # For each controlled joint j:
        #
        #       e(q + delta)
        #       e(q - delta)
        #
        # J[:, j] =
        #
        #   [e(q+dq)-e(q-dq)] / (2*dq)
        #
        # Units:
        #
        # px / degree
        # degree_error / degree
        # -----------------------------------------------------

        jacobian_columns: list[
            np.ndarray
        ] = []

        measurement_records = []

        delta = float(
            args.delta_deg
        )

        for column_index, joint_index in (
            enumerate(
                controlled_joints
            )
        ):
            print()
            print(
                "----------------------------------------"
            )

            print(
                f"Calibrating joint "
                f"{joint_index} "
                f"({column_index + 1}/"
                f"{len(controlled_joints)})"
            )

            print(
                "----------------------------------------"
            )

            plus_q = (
                baseline_q.copy()
            )

            minus_q = (
                baseline_q.copy()
            )

            plus_q[
                joint_index
            ] += delta

            minus_q[
                joint_index
            ] -= delta

            error_plus = None
            error_minus = None

            try:
                # ---------------------------------------------
                # +delta
                # ---------------------------------------------

                print(
                    f"Move q[{joint_index}] "
                    f"to +{delta:.4f} deg"
                )

                motion.move_to(
                    target_q=plus_q,
                    duration_s=(
                        args.move_duration_s
                    ),
                    realtime=True,
                )

                if args.settle_s > 0:
                    time.sleep(
                        args.settle_s
                    )

                error_plus = (
                    measure_mean_error(
                        robot=robot,
                        estimator=estimator,
                        error_keys=error_keys,
                        samples=args.samples,
                        sample_interval_s=(
                            args
                            .sample_interval_s
                        ),
                    )
                )

                print_error_vector(
                    "+delta error:",
                    error_keys,
                    error_plus,
                )

                # ---------------------------------------------
                # -delta
                #
                # This movement spans 2*delta relative to the
                # previous +delta pose.
                # Planner/ActionGuard still enforce limits.
                # ---------------------------------------------

                print()
                print(
                    f"Move q[{joint_index}] "
                    f"to -{delta:.4f} deg"
                )

                motion.move_to(
                    target_q=minus_q,
                    duration_s=(
                        args.move_duration_s
                        * 2.0
                    ),
                    realtime=True,
                )

                if args.settle_s > 0:
                    time.sleep(
                        args.settle_s
                    )

                error_minus = (
                    measure_mean_error(
                        robot=robot,
                        estimator=estimator,
                        error_keys=error_keys,
                        samples=args.samples,
                        sample_interval_s=(
                            args
                            .sample_interval_s
                        ),
                    )
                )

                print_error_vector(
                    "-delta error:",
                    error_keys,
                    error_minus,
                )

                column = (
                    error_plus
                    - error_minus
                ) / (
                    2.0 * delta
                )

                if not np.all(
                    np.isfinite(
                        column
                    )
                ):
                    raise ValueError(
                        "Jacobian column contains "
                        "NaN or Inf"
                    )

                jacobian_columns.append(
                    column
                )

                measurement_records.append(
                    {
                        "joint_index": int(
                            joint_index
                        ),
                        "error_plus": (
                            error_plus.tolist()
                        ),
                        "error_minus": (
                            error_minus.tolist()
                        ),
                        "jacobian_column": (
                            column.tolist()
                        ),
                    }
                )

                print_error_vector(
                    "Jacobian column:",
                    error_keys,
                    column,
                )

            finally:
                # ---------------------------------------------
                # Always return to baseline before calibrating
                # the next joint.
                # ---------------------------------------------

                print()
                print(
                    "Returning to baseline..."
                )

                motion.move_to(
                    target_q=baseline_q,
                    duration_s=(
                        args.return_duration_s
                    ),
                    realtime=True,
                )

                if args.settle_s > 0:
                    time.sleep(
                        args.settle_s
                    )

        # -----------------------------------------------------
        # Build complete J matrix.
        # -----------------------------------------------------

        jacobian = np.stack(
            jacobian_columns,
            axis=1,
        )

        expected_shape = (
            len(error_keys),
            len(controlled_joints),
        )

        if (
            jacobian.shape
            != expected_shape
        ):
            raise RuntimeError(
                "Unexpected Jacobian shape: "
                f"{jacobian.shape}, "
                f"expected {expected_shape}"
            )

        if not np.all(
            np.isfinite(
                jacobian
            )
        ):
            raise ValueError(
                "Image Jacobian contains "
                "NaN or Inf"
            )

        # -----------------------------------------------------
        # Numerical diagnostics.
        # -----------------------------------------------------

        rank = int(
            np.linalg.matrix_rank(
                jacobian
            )
        )

        singular_values = (
            np.linalg.svd(
                jacobian,
                compute_uv=False,
            )
        )

        if (
            singular_values.size > 0
            and singular_values[-1] > 0
        ):
            condition_number = float(
                singular_values[0]
                / singular_values[-1]
            )
        else:
            condition_number = float(
                "inf"
            )

        print()
        print(
            "========================================"
        )

        print(
            "IMAGE JACOBIAN"
        )

        print(
            "========================================"
        )

        print(
            jacobian
        )

        print()
        print(
            f"shape            : "
            f"{jacobian.shape}"
        )

        print(
            f"rank             : "
            f"{rank}"
        )

        print(
            f"singular values  : "
            f"{singular_values}"
        )

        print(
            f"condition number : "
            f"{condition_number}"
        )

        # -----------------------------------------------------
        # Save results.
        # -----------------------------------------------------

        output_path = (
            vision
            .servo
            .jacobian_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            jacobian,
            allow_pickle=False,
        )

        metadata_path = (
            output_path
            .with_suffix(
                ".json"
            )
        )

        metadata = {
            "error_keys": list(
                error_keys
            ),

            "controlled_joint_indices": [
                int(value)
                for value
                in controlled_joints
            ],

            "baseline_q": (
                baseline_q.tolist()
            ),

            "baseline_error": (
                baseline_error.tolist()
            ),

            "delta_deg": float(
                args.delta_deg
            ),

            "samples": int(
                args.samples
            ),

            "sample_interval_s": float(
                args.sample_interval_s
            ),

            "settle_s": float(
                args.settle_s
            ),

            "move_duration_s": float(
                args.move_duration_s
            ),

            "return_duration_s": float(
                args.return_duration_s
            ),

            "max_step_deg": float(
                args.max_step_deg
            ),

            "max_velocity_deg_s": float(
                args.max_velocity_deg_s
            ),

            "max_acceleration_deg_s2": float(
                args.max_acceleration_deg_s2
            ),

            "jacobian_shape": list(
                jacobian.shape
            ),

            "jacobian_rank": rank,

            "singular_values": (
                singular_values.tolist()
            ),

            "condition_number": (
                condition_number
            ),

            "measurements": (
                measurement_records
            ),
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        calibration_completed = True

        print()
        print(
            "Saved Jacobian:"
        )

        print(
            output_path
        )

        print()
        print(
            "Saved metadata:"
        )

        print(
            metadata_path
        )

    finally:
        # -----------------------------------------------------
        # Final safety return.
        #
        # Even if calibration fails halfway through,
        # try to restore the baseline pose before disconnect.
        # -----------------------------------------------------

        if (
            connected
            and motion is not None
            and baseline_q is not None
        ):
            try:
                print()
                print(
                    "Final return to baseline..."
                )

                motion.move_to(
                    target_q=baseline_q,
                    duration_s=max(
                        args.return_duration_s,
                        1.5,
                    ),
                    realtime=True,
                )

                print(
                    "Baseline restored."
                )

            except Exception as exc:
                print()
                print(
                    "WARNING:"
                )

                print(
                    "Failed to restore baseline:"
                )

                print(
                    repr(exc)
                )

        if connected:
            try:
                print()
                print(
                    "Disconnecting SO-101..."
                )

                robot.close()

                print(
                    "Disconnected."
                )

            except Exception as exc:
                print(
                    "WARNING: disconnect failed:"
                )

                print(
                    repr(exc)
                )

    if calibration_completed:
        print()
        print(
            "========================================"
        )

        print(
            "JACOBIAN CALIBRATION COMPLETE"
        )

        print(
            "========================================"
        )


if __name__ == "__main__":
    main()
