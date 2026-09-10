from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from usb_insertion.control.action_guard import (
    ActionGuard,
    ActionGuardConfig,
)
from usb_insertion.core.datatypes import (
    Action,
)
from usb_insertion.fakes.fake_alignment_estimator import (
    FakeAlignmentEstimator,
)
from usb_insertion.fakes.fake_cameras import (
    FakeCameras,
)
from usb_insertion.fakes.fake_observation_source import (
    FakeObservationSource,
)
from usb_insertion.fakes.fake_policy import (
    FakePolicy,
)
from usb_insertion.fakes.fake_robot import (
    FakeRobot,
)
from usb_insertion.fakes.fake_success_detector import (
    FakeSuccessDetector,
)
from usb_insertion.fakes.fake_visual_insertion import (
    FakeVisualInsertionController,
)
from usb_insertion.fakes.fake_visual_servo import (
    FakeVisualServoController,
)
from usb_insertion.runtime.jsonl_logger import (
    JsonlLogger,
)
from usb_insertion.workflow.orchestrator import (
    OrchestratorConfig,
    TaskOrchestrator,
    TaskRunResult,
)
from usb_insertion.workflow.state_machine import (
    TaskStateMachine,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_dry_run(
    log_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[TaskRunResult, Path]:

    home_q = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.25,
    ])

    # ACT 将机器人带到这里附近。
    capture_q = np.array([
        0.50,
        0.00,
        0.00,
        0.00,
        0.00,
        0.25,
    ])

    # Visual Servo 最终对准状态。
    aligned_q = np.array([
        0.50,
        0.20,
        -0.10,
        0.10,
        0.00,
        0.25,
    ])

    if log_path is None:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        log_path = (
            _project_root()
            / "logs"
            / f"dry_run_{timestamp}.jsonl"
        )

    log_path = Path(log_path)

    logger = JsonlLogger(
        log_path
    )

    robot = FakeRobot(
        dof=6,
        initial_q=home_q,
    )

    cameras = FakeCameras()

    observation_source = (
        FakeObservationSource(
            robot=robot,
            cameras=cameras,
        )
    )

    # ACT：
    # 模拟把 USB 从随机区抓起并搬到
    # Visual Servo capture region。
    policy = FakePolicy(
        target_q=capture_q,
        gain=0.5,
    )

    estimator = (
        FakeAlignmentEstimator(
            capture_target_q=capture_q,
            aligned_target_q=aligned_q,
            capture_tolerance=0.05,
            aligned_tolerance=0.02,

            # index 0 在 Fake 世界里作为插入轴。
            # alignment 不检查它。
            alignment_indices=(
                1,
                2,
                3,
                4,
            ),
        )
    )

    visual_servo = (
        FakeVisualServoController(
            aligned_target_q=aligned_q,
            gain=0.5,
        )
    )

    insertion_controller = (
        FakeVisualInsertionController(
            axis_index=0,
            target_position=0.80,
            step_size=0.08,
        )
    )

    # 第一次插入失败；
    # 第二次第 3 个 detector check 成功。
    detector = FakeSuccessDetector(
        attempt_schedule=[
            None,
            3,
        ]
    )

    action_guard = ActionGuard(
        ActionGuardConfig(
            gripper_index=5,
            fixed_gripper_position=0.25,
        )
    )

    def move_to_pose(
        name: str,
        target_q: np.ndarray,
    ) -> None:
        if verbose:
            print(
                f"[MOTION] {name}"
            )

        current_q = (
            robot.get_state().q
        )

        safe_action = (
            action_guard.apply(
                raw_action=Action(
                    q=target_q
                ),
                current_q=current_q,
            )
        )

        robot.send_action(
            safe_action
        )

        logger.write(
            "motion",
            name=name,
            final_q=(
                robot.get_state().q
            ),
        )

    def home_motion() -> None:
        move_to_pose(
            "home",
            home_q,
        )

    def retract_motion() -> None:
        move_to_pose(
            "retract_to_preinsert",
            aligned_q,
        )

    orchestrator = TaskOrchestrator(
        state_machine=TaskStateMachine(
            max_attempts=2
        ),
        observation_source=(
            observation_source
        ),
        policy=policy,
        alignment_estimator=estimator,
        visual_servo=visual_servo,
        insertion_controller=(
            insertion_controller
        ),
        detector=detector,
        robot=robot,
        action_guard=action_guard,
        home_motion=home_motion,
        retract_motion=retract_motion,
        config=OrchestratorConfig(
            control_fps=4,
            act_approach_timeout_s=3.0,
            visual_alignment_timeout_s=3.0,
            insertion_timeout_s=1.0,
        ),
        logger=logger,
    )

    robot.connect()
    cameras.open()

    logger.write(
        "dry_run_started"
    )

    try:
        result = orchestrator.run()

        logger.write(
            "dry_run_finished",
            success=result.success,
            attempts=result.attempts,
        )

    finally:
        cameras.close()
        robot.close()

    if verbose:
        print()
        print(
            "========== DRY RUN RESULT =========="
        )

        print(
            f"success          : {result.success}"
        )

        print(
            f"final_state      : "
            f"{result.final_state.name}"
        )

        print(
            f"attempts         : "
            f"{result.attempts}"
        )

        print(
            f"ACT steps        : "
            f"{result.act_policy_steps}"
        )

        print(
            f"alignment steps  : "
            f"{result.alignment_steps}"
        )

        print(
            f"insertion steps  : "
            f"{result.insertion_steps}"
        )

        print(
            f"log              : {log_path}"
        )

        print(
            "===================================="
        )

    return result, log_path


def main() -> None:
    run_dry_run()


if __name__ == "__main__":
    main()