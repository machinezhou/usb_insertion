from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from usb_insertion.adapters.policy.lerobot_act import (
    LeRobotACTAdapter,
)
from usb_insertion.adapters.robot.factory import (
    build_so101_adapter,
)
from usb_insertion.config.hardware import (
    load_hardware_config,
)
from usb_insertion.config.run import (
    load_run_config,
)
from usb_insertion.control.action_guard import (
    ActionGuard,
    ActionGuardConfig,
)
from usb_insertion.control.scripted_motion import (
    ScriptedMotionController,
)
from usb_insertion.runtime.jsonl_logger import (
    JsonlLogger,
)
from usb_insertion.runtime.preflight import (
    run_preflight,
)
from usb_insertion.runtime.rate import (
    RateLimiter,
)
from usb_insertion.vision.alignment.config import (
    VisualServoConfig,
    load_vision_alignment_config,
)
from usb_insertion.vision.alignment.estimator import (
    TemplateAlignmentEstimator,
)
from usb_insertion.vision.alignment.face import (
    FaceAwareAlignmentEstimator,
    TemplatePlugFaceClassifier,
    load_plug_face_config,
)
from usb_insertion.vision.alignment.servo import (
    JacobianVisualServoController,
)
from usb_insertion.vision.insertion.controller import (
    CalibratedVisualInsertionController,
)
from usb_insertion.vision.insertion.success_detector import (
    AlignmentSuccessDetector,
    load_success_reference,
)
from usb_insertion.workflow.orchestrator import (
    OrchestratorConfig,
    TaskOrchestrator,
)
from usb_insertion.workflow.state_machine import (
    TaskStateMachine,
)


def project_root() -> Path:
    return Path(
        __file__
    ).resolve().parents[3]


def resolve_path(
    root: Path,
    value: str,
) -> Path:

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = root / path

    return path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hardware",
        default=(
            "configs/hardware.yaml"
        ),
    )

    parser.add_argument(
        "--vision",
        default=(
            "configs/vision.yaml"
        ),
    )

    parser.add_argument(
        "--face",
        default=(
            "configs/plug_face.yaml"
        ),
    )

    parser.add_argument(
        "--run",
        default=(
            "configs/run.yaml"
        ),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    parser.add_argument(
        "--enable-motion",
        action="store_true",
    )

    args = parser.parse_args()

    root = project_root()

    hardware = (
        load_hardware_config(
            resolve_path(
                root,
                args.hardware,
            )
        )
    )

    vision = (
        load_vision_alignment_config(
            resolve_path(
                root,
                args.vision,
            )
        )
    )

    face_config = (
        load_plug_face_config(
            resolve_path(
                root,
                args.face,
            )
        )
    )

    run_cfg = load_run_config(
        resolve_path(
            root,
            args.run,
        )
    )

    report = run_preflight(
        hardware=hardware,
        vision=vision,
        face=face_config,
        run_cfg=run_cfg,
        check_devices=True,
    )

    print(
        "========== PREFLIGHT =========="
    )

    print(
        report.format_text()
    )

    print(
        "==============================="
    )

    report.raise_if_failed()

    if args.check_only:
        print()
        print(
            "Preflight passed."
        )

        print(
            "No robot connection "
            "or motion was performed."
        )

        return

    if not args.enable_motion:
        raise RuntimeError(
            "Real rollout requires "
            "--enable-motion."
        )

    policy = (
        LeRobotACTAdapter
        .from_pretrained(
            checkpoint=(
                run_cfg
                .act
                .checkpoint
            ),
            dataset_repo_id=(
                run_cfg
                .act
                .dataset_repo_id
            ),
            dataset_root=(
                run_cfg
                .act
                .dataset_root
            ),
            device=(
                run_cfg
                .act
                .device
            ),
            task=(
                run_cfg
                .act
                .task
            ),
        )
    )

    geometric_estimator = (
        TemplateAlignmentEstimator(
            vision
        )
    )

    face_classifier = (
        TemplatePlugFaceClassifier(
            face_config
        )
    )

    estimator = (
        FaceAwareAlignmentEstimator(
            geometric_estimator=(
                geometric_estimator
            ),
            face_classifier=(
                face_classifier
            ),
        )
    )

    alignment_servo = (
        JacobianVisualServoController(
            vision.servo
        )
    )

    insertion_servo_config = (
        VisualServoConfig(
            error_keys=(
                run_cfg
                .insertion
                .correction_error_keys
            ),
            controlled_joint_indices=(
                vision.servo
                .controlled_joint_indices
            ),
            jacobian_path=(
                vision.servo
                .jacobian_path
            ),
            gain=(
                run_cfg
                .insertion
                .correction_gain
            ),
            damping=(
                run_cfg
                .insertion
                .correction_damping
            ),
            max_joint_delta=(
                run_cfg
                .insertion
                .correction_max_joint_delta
            ),
        )
    )

    insertion_servo = (
        JacobianVisualServoController(
            insertion_servo_config,
            jacobian_error_keys=(
                vision.servo
                .error_keys
            ),
            jacobian_controlled_joint_indices=(
                vision.servo
                .controlled_joint_indices
            ),
        )
    )

    total_joint_delta = (
        np.asarray(
            np.load(
                run_cfg
                .insertion
                .joint_delta_path,
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
    )

    insertion_controller = (
        CalibratedVisualInsertionController(
            total_joint_delta=(
                total_joint_delta
            ),
            progress_step=(
                run_cfg
                .insertion
                .progress_step
            ),
            correction_servo=(
                insertion_servo
            ),
            guard_error_limits=(
                run_cfg
                .insertion
                .guard_error_limits
            ),
        )
    )

    success_detector = (
        AlignmentSuccessDetector(
            load_success_reference(
                run_cfg
                .insertion
                .success_reference_path
            )
        )
    )

    robot = (
        build_so101_adapter(
            hardware
        )
    )

    # 注意：
    # ACT 阶段必须能控制 gripper，
    # 因此这里不能全局固定 gripper。
    action_guard = (
        ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=(
                    np.asarray(
                        run_cfg
                        .max_delta_per_step,
                        dtype=np.float64,
                    )
                ),
            )
        )
    )

    scripted_motion = (
        ScriptedMotionController(
            robot=robot,
            action_guard=(
                action_guard
            ),
            control_fps=(
                run_cfg.control_fps
            ),
        )
    )

    home_q = np.asarray(
        run_cfg.home_q,
        dtype=np.float64,
    )

    def home_motion():
        scripted_motion.move_to(
            target_q=home_q,
            duration_s=(
                run_cfg
                .home_duration_s
            ),
            realtime=True,
        )

    def retract_motion():
        current_q = (
            robot.get_state().q
        )

        retract_q = (
            current_q
            - total_joint_delta
            * run_cfg
            .insertion
            .retract_scale
        )

        # Retry 阶段绝不能改变
        # ACT 已经抓住 USB 的夹爪位置。
        retract_q[5] = (
            current_q[5]
        )

        scripted_motion.move_to(
            target_q=retract_q,
            duration_s=(
                run_cfg
                .insertion
                .retract_duration_s
            ),
            realtime=True,
        )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    logger = JsonlLogger(
        root
        / "logs"
        / (
            f"real_task_"
            f"{timestamp}.jsonl"
        )
    )

    rate = RateLimiter(
        run_cfg.control_fps
    )

    orchestrator = (
        TaskOrchestrator(
            state_machine=(
                TaskStateMachine(
                    max_attempts=(
                        run_cfg
                        .max_attempts
                    )
                )
            ),
            observation_source=(
                robot
            ),
            policy=policy,
            alignment_estimator=(
                estimator
            ),
            visual_servo=(
                alignment_servo
            ),
            insertion_controller=(
                insertion_controller
            ),
            detector=(
                success_detector
            ),
            robot=robot,
            action_guard=(
                action_guard
            ),
            home_motion=(
                home_motion
            ),
            retract_motion=(
                retract_motion
            ),
            config=(
                OrchestratorConfig(
                    control_fps=(
                        run_cfg
                        .control_fps
                    ),
                    act_approach_timeout_s=(
                        run_cfg
                        .act
                        .timeout_s
                    ),
                    visual_alignment_timeout_s=(
                        run_cfg
                        .visual_alignment_timeout_s
                    ),
                    insertion_timeout_s=(
                        run_cfg
                        .insertion
                        .timeout_s
                    ),
                )
            ),
            logger=logger,
            step_wait=(
                rate.wait
            ),
        )
    )

    print(
        "Connecting SO-101..."
    )

    connected = False

    try:
        robot.connect()
        connected = True

        rate.reset()

        result = (
            orchestrator.run()
        )

        print()
        print(
            "========== TASK RESULT =========="
        )

        print(
            "success:",
            result.success,
        )

        print(
            "state:",
            result.final_state.name,
        )

        print(
            "attempts:",
            result.attempts,
        )

        print(
            "ACT steps:",
            result.act_policy_steps,
        )

        print(
            "alignment steps:",
            result.alignment_steps,
        )

        print(
            "insertion steps:",
            result.insertion_steps,
        )

        print(
            "================================="
        )

    finally:
        if connected:
            robot.close()


if __name__ == "__main__":
    main()
