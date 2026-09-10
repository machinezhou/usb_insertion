import unittest

import numpy as np

from usb_insertion.control.action_guard import (
    ActionGuard,
    ActionGuardConfig,
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
from usb_insertion.workflow.orchestrator import (
    OrchestratorConfig,
    TaskOrchestrator,
)
from usb_insertion.workflow.state_machine import (
    TaskStateMachine,
)
from usb_insertion.workflow.states import (
    TaskState,
)


class MotionCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


class TestTaskOrchestrator(
    unittest.TestCase
):
    def test_full_pipeline_retry_then_success(
        self,
    ):
        home_q = np.array([
            0,
            0,
            0,
            0,
            0,
            0.25,
        ])

        capture_q = np.array([
            0.5,
            0,
            0,
            0,
            0,
            0.25,
        ])

        aligned_q = np.array([
            0.5,
            0.2,
            -0.1,
            0.1,
            0,
            0.25,
        ])

        robot = FakeRobot(
            dof=6,
            initial_q=home_q,
        )

        cameras = FakeCameras()

        robot.connect()
        cameras.open()

        source = FakeObservationSource(
            robot=robot,
            cameras=cameras,
        )

        guard = ActionGuard(
            ActionGuardConfig(
                gripper_index=5,
                fixed_gripper_position=0.25,
            )
        )

        home = MotionCounter()

        def retract():
            robot.send_action(
                guard.apply(
                    raw_action=(
                        __import__(
                            "usb_insertion.core.datatypes",
                            fromlist=["Action"],
                        ).Action(
                            q=aligned_q
                        )
                    ),
                    current_q=(
                        robot.get_state().q
                    ),
                )
            )

        orchestrator = TaskOrchestrator(
            state_machine=(
                TaskStateMachine(
                    max_attempts=2
                )
            ),
            observation_source=source,
            policy=FakePolicy(
                target_q=capture_q,
                gain=0.5,
            ),
            alignment_estimator=(
                FakeAlignmentEstimator(
                    capture_target_q=(
                        capture_q
                    ),
                    aligned_target_q=(
                        aligned_q
                    ),
                    capture_tolerance=0.05,
                    aligned_tolerance=0.02,
                    alignment_indices=(
                        1,
                        2,
                        3,
                        4,
                    ),
                )
            ),
            visual_servo=(
                FakeVisualServoController(
                    aligned_target_q=(
                        aligned_q
                    ),
                    gain=0.5,
                )
            ),
            insertion_controller=(
                FakeVisualInsertionController(
                    axis_index=0,
                    target_position=0.8,
                    step_size=0.08,
                )
            ),
            detector=(
                FakeSuccessDetector(
                    attempt_schedule=[
                        None,
                        3,
                    ]
                )
            ),
            robot=robot,
            action_guard=guard,
            home_motion=home,
            retract_motion=retract,
            config=OrchestratorConfig(
                control_fps=4,
                act_approach_timeout_s=3,
                visual_alignment_timeout_s=3,
                insertion_timeout_s=1,
            ),
        )

        try:
            result = orchestrator.run()
        finally:
            cameras.close()
            robot.close()

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.final_state,
            TaskState.DONE,
        )

        self.assertEqual(
            result.attempts,
            2,
        )

        self.assertGreater(
            result.act_policy_steps,
            0,
        )

        self.assertGreater(
            result.alignment_steps,
            0,
        )

        self.assertGreater(
            result.insertion_steps,
            0,
        )

        self.assertEqual(
            home.calls,
            1,
        )

    def test_invalid_config(self):
        with self.assertRaises(
            ValueError
        ):
            OrchestratorConfig(
                control_fps=0
            )


if __name__ == "__main__":
    unittest.main()