from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from usb_insertion.control.action_guard import (
    ActionGuard,
)
from usb_insertion.core.datatypes import (
    Action,
    DetectionResult,
)
from usb_insertion.core.protocols import (
    AlignmentEstimator,
    ObservationSource,
    PolicyBackend,
    RobotBackend,
    SuccessDetector,
    VisualInsertionController,
    VisualServoController,
)
from usb_insertion.runtime.control_loop import (
    run_policy_step_from_observation,
)
from usb_insertion.runtime.jsonl_logger import (
    JsonlLogger,
)
from usb_insertion.workflow.state_machine import (
    TaskStateMachine,
)
from usb_insertion.workflow.states import (
    TaskEvent,
    TaskState,
)


MotionCommand = Callable[
    [],
    None,
]

StepWait = Callable[
    [],
    None,
]


@dataclass(slots=True)
class OrchestratorConfig:
    control_fps: float = 30.0

    act_approach_timeout_s: float = 12.0
    visual_alignment_timeout_s: float = 8.0
    insertion_timeout_s: float = 8.0

    def __post_init__(self) -> None:
        for name, value in (
            (
                "control_fps",
                self.control_fps,
            ),
            (
                "act_approach_timeout_s",
                self.act_approach_timeout_s,
            ),
            (
                "visual_alignment_timeout_s",
                self.visual_alignment_timeout_s,
            ),
            (
                "insertion_timeout_s",
                self.insertion_timeout_s,
            ),
        ):
            if value <= 0:
                raise ValueError(
                    f"{name} must be > 0"
                )

    def _steps(
        self,
        timeout_s: float,
    ) -> int:
        return max(
            1,
            math.ceil(
                timeout_s
                * self.control_fps
            ),
        )

    @property
    def act_max_steps(self) -> int:
        return self._steps(
            self.act_approach_timeout_s
        )

    @property
    def alignment_max_steps(self) -> int:
        return self._steps(
            self.visual_alignment_timeout_s
        )

    @property
    def insertion_max_steps(self) -> int:
        return self._steps(
            self.insertion_timeout_s
        )


@dataclass(slots=True)
class TaskRunResult:
    final_state: TaskState
    attempts: int

    act_policy_steps: int
    alignment_steps: int
    insertion_steps: int

    @property
    def success(self) -> bool:
        return (
            self.final_state
            is TaskState.DONE
        )


class TaskOrchestrator:
    def __init__(
        self,
        state_machine: TaskStateMachine,
        observation_source: ObservationSource,
        policy: PolicyBackend,
        alignment_estimator: AlignmentEstimator,
        visual_servo: VisualServoController,
        insertion_controller: VisualInsertionController,
        detector: SuccessDetector,
        robot: RobotBackend,
        action_guard: ActionGuard,
        home_motion: MotionCommand,
        retract_motion: MotionCommand,
        config: OrchestratorConfig,
        logger: JsonlLogger | None = None,
        step_wait: StepWait | None = None,
    ):
        self._state_machine = (
            state_machine
        )

        self._observation_source = (
            observation_source
        )

        self._policy = policy

        self._alignment_estimator = (
            alignment_estimator
        )

        self._visual_servo = (
            visual_servo
        )

        self._insertion_controller = (
            insertion_controller
        )

        self._detector = detector
        self._robot = robot

        self._action_guard = (
            action_guard
        )

        self._home_motion = (
            home_motion
        )

        self._retract_motion = (
            retract_motion
        )

        self._config = config
        self._logger = logger

        self._step_wait = (
            step_wait
            if step_wait is not None
            else lambda: None
        )

        self._act_policy_steps = 0
        self._alignment_steps = 0
        self._insertion_steps = 0

    def run(self) -> TaskRunResult:
        if (
            self._state_machine.state
            is not TaskState.INIT
        ):
            raise RuntimeError(
                "State machine must "
                "start in INIT"
            )

        self._act_policy_steps = 0
        self._alignment_steps = 0
        self._insertion_steps = 0

        self._policy.reset()
        self._alignment_estimator.reset()
        self._visual_servo.reset()
        self._insertion_controller.reset()

        self._dispatch(
            TaskEvent.START
        )

        self._home_motion()

        self._dispatch(
            TaskEvent.HOME_REACHED
        )

        while not (
            self._state_machine.is_terminal
        ):
            state = (
                self._state_machine.state
            )

            if (
                state
                is TaskState.ACT_PICK_APPROACH
            ):
                reached = (
                    self._run_act_approach()
                )

                self._dispatch(
                    TaskEvent.ACT_CAPTURE_REACHED
                    if reached
                    else TaskEvent.ACT_APPROACH_FAILED
                )

            elif (
                state
                is TaskState.VISUAL_ALIGNING
            ):
                aligned = (
                    self._run_visual_alignment()
                )

                self._dispatch(
                    TaskEvent.VISUAL_ALIGNED
                    if aligned
                    else TaskEvent.VISUAL_ALIGNMENT_FAILED
                )

            elif (
                state
                is TaskState.PREINSERT
            ):
                self._detector.reset()

                self._insertion_controller.reset()

                self._dispatch(
                    TaskEvent.START_INSERTION
                )

            elif (
                state
                is TaskState.VISUAL_INSERTING
            ):
                detection = (
                    self._run_visual_insertion()
                )

                self._dispatch(
                    TaskEvent.INSERTION_FINISHED
                )

                self._dispatch(
                    TaskEvent.VERIFY_SUCCESS
                    if detection.success
                    else TaskEvent.VERIFY_FAILURE
                )

            elif (
                state
                is TaskState.RETRACTING
            ):
                self._retract_motion()

                self._alignment_estimator.reset()
                self._visual_servo.reset()

                self._dispatch(
                    TaskEvent.RETRACT_FINISHED
                )

            else:
                raise RuntimeError(
                    "Unexpected state: "
                    f"{state.name}"
                )

        result = TaskRunResult(
            final_state=(
                self._state_machine.state
            ),
            attempts=(
                self._state_machine.attempts
            ),
            act_policy_steps=(
                self._act_policy_steps
            ),
            alignment_steps=(
                self._alignment_steps
            ),
            insertion_steps=(
                self._insertion_steps
            ),
        )

        self._log(
            "task_finished",
            final_state=result.final_state,
            success=result.success,
            attempts=result.attempts,
            act_policy_steps=(
                result.act_policy_steps
            ),
            alignment_steps=(
                result.alignment_steps
            ),
            insertion_steps=(
                result.insertion_steps
            ),
        )

        return result

    def _run_act_approach(
        self,
    ) -> bool:
        for step_index in range(
            1,
            self._config.act_max_steps
            + 1,
        ):
            observation = (
                self._observation_source
                .get_observation()
            )

            alignment = (
                self._alignment_estimator
                .estimate(
                    observation
                )
            )

            self._log(
                "act_capture_check",
                step=step_index,
                alignment=alignment,
            )

            if (
                alignment.visible
                and alignment
                .in_capture_region
            ):
                return True

            result = (
                run_policy_step_from_observation(
                    observation=observation,
                    policy=self._policy,
                    robot=self._robot,
                    action_guard=(
                        self._action_guard
                    ),
                )
            )

            self._act_policy_steps += 1

            self._log(
                "act_control_step",
                step=step_index,
                raw_action=(
                    result.raw_action
                ),
                safe_action=(
                    result.safe_action
                ),
            )

            self._step_wait()

        observation = (
            self._observation_source
            .get_observation()
        )

        alignment = (
            self._alignment_estimator
            .estimate(
                observation
            )
        )

        return (
            alignment.visible
            and alignment
            .in_capture_region
        )

    def _run_visual_alignment(
        self,
    ) -> bool:
        for step_index in range(
            1,
            self._config
            .alignment_max_steps
            + 1,
        ):
            observation = (
                self._observation_source
                .get_observation()
            )

            alignment = (
                self._alignment_estimator
                .estimate(
                    observation
                )
            )

            self._log(
                "visual_alignment_check",
                step=step_index,
                alignment=alignment,
            )

            if (
                alignment.visible
                and alignment.aligned
            ):
                return True

            if not alignment.visible:
                self._step_wait()
                continue

            raw_action = (
                self._visual_servo
                .compute_action(
                    observation,
                    alignment,
                )
            )

            safe_action = (
                self._send_guarded(
                    raw_action,
                    observation.robot.q,
                )
            )

            self._alignment_steps += 1

            self._log(
                "visual_alignment_step",
                step=step_index,
                raw_action=raw_action,
                safe_action=safe_action,
                alignment=alignment,
            )

            self._step_wait()

        final_observation = (
            self._observation_source
            .get_observation()
        )

        final_alignment = (
            self._alignment_estimator
            .estimate(
                final_observation
            )
        )

        return (
            final_alignment.visible
            and final_alignment.aligned
        )

    def _run_visual_insertion(
        self,
    ) -> DetectionResult:

        for step_index in range(
            1,
            self._config
            .insertion_max_steps
            + 1,
        ):
            observation = (
                self._observation_source
                .get_observation()
            )

            alignment = (
                self._alignment_estimator
                .estimate(
                    observation
                )
            )

            detection = (
                self._detector.update(
                    observation.cameras,
                    alignment,
                )
            )

            self._log(
                "verification",
                step=step_index,
                attempt=(
                    self._state_machine
                    .attempts
                ),
                detection=detection,
                alignment=alignment,
            )

            if detection.success:
                return detection

            if not (
                self._insertion_controller
                .is_safe(
                    alignment
                )
            ):
                return DetectionResult(
                    success=False,
                    reason=(
                        "insertion_visual_guard"
                    ),
                )

            raw_action = (
                self._insertion_controller
                .compute_action(
                    observation,
                    alignment,
                )
            )

            safe_action = (
                self._send_guarded(
                    raw_action,
                    observation.robot.q,
                )
            )

            self._insertion_steps += 1

            self._log(
                "visual_insertion_step",
                step=step_index,
                attempt=(
                    self._state_machine
                    .attempts
                ),
                raw_action=raw_action,
                safe_action=safe_action,
                alignment=alignment,
            )

            self._step_wait()

        final_observation = (
            self._observation_source
            .get_observation()
        )

        final_alignment = (
            self._alignment_estimator
            .estimate(
                final_observation
            )
        )

        final_detection = (
            self._detector.update(
                final_observation.cameras,
                final_alignment,
            )
        )

        if final_detection.success:
            return final_detection

        return DetectionResult(
            success=False,
            score=final_detection.score,
            consecutive=(
                final_detection.consecutive
            ),
            reason="insertion_timeout",
        )

    def _send_guarded(
        self,
        raw_action: Action,
        current_q,
    ) -> Action:
        safe_action = (
            self._action_guard.apply(
                raw_action=raw_action,
                current_q=current_q,
            )
        )

        self._robot.send_action(
            safe_action
        )

        return safe_action

    def _dispatch(
        self,
        event: TaskEvent,
    ) -> None:
        old_state = (
            self._state_machine.state
        )

        new_state = (
            self._state_machine.dispatch(
                event
            )
        )

        self._log(
            "state_transition",
            from_state=old_state,
            task_event=event,
            to_state=new_state,
            attempts=(
                self._state_machine.attempts
            ),
        )

    def _log(
        self,
        event: str,
        **payload,
    ) -> None:
        if self._logger is not None:
            self._logger.write(
                event,
                **payload,
            )
