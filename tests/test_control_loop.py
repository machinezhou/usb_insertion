import unittest

import numpy as np

from usb_insertion.control.action_guard import (
    ActionGuard,
    ActionGuardConfig,
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
from usb_insertion.runtime.control_loop import (
    ControlStepResult,
    run_policy_step,
    run_policy_step_from_observation,
)


class TestControlLoop(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot(
            dof=6
        )

        self.cameras = FakeCameras()

        self.robot.connect()
        self.cameras.open()

        self.observation_source = (
            FakeObservationSource(
                robot=self.robot,
                cameras=self.cameras,
            )
        )

    def tearDown(self):
        self.cameras.close()
        self.robot.close()

    def test_policy_step_returns_result(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.5,
        )

        guard = ActionGuard(
            ActionGuardConfig()
        )

        result = run_policy_step(
            observation_source=(
                self.observation_source
            ),
            policy=policy,
            robot=self.robot,
            action_guard=guard,
        )

        self.assertIsInstance(
            result,
            ControlStepResult,
        )

    def test_policy_step_updates_robot(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.5,
        )

        guard = ActionGuard(
            ActionGuardConfig()
        )

        run_policy_step(
            observation_source=(
                self.observation_source
            ),
            policy=policy,
            robot=self.robot,
            action_guard=guard,
        )

        state = self.robot.get_state()

        np.testing.assert_allclose(
            state.q,
            np.full(6, 0.5),
        )

    def test_raw_action_is_preserved(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.5,
        )

        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=np.full(
                    6,
                    0.1,
                )
            )
        )

        result = run_policy_step(
            observation_source=(
                self.observation_source
            ),
            policy=policy,
            robot=self.robot,
            action_guard=guard,
        )

        # FakePolicy 原始输出：
        # 0 + 0.5 * (1 - 0) = 0.5
        np.testing.assert_allclose(
            result.raw_action.q,
            np.full(6, 0.5),
        )

    def test_action_guard_limits_robot_action(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.5,
        )

        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=np.full(
                    6,
                    0.1,
                )
            )
        )

        result = run_policy_step(
            observation_source=(
                self.observation_source
            ),
            policy=policy,
            robot=self.robot,
            action_guard=guard,
        )

        np.testing.assert_allclose(
            result.safe_action.q,
            np.full(6, 0.1),
        )

        np.testing.assert_allclose(
            self.robot.get_state().q,
            np.full(6, 0.1),
        )

    def test_multiple_control_steps_form_closed_loop(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.5,
        )

        guard = ActionGuard(
            ActionGuardConfig()
        )

        # Step 1:
        # 0 -> 0.5
        run_policy_step(
            self.observation_source,
            policy,
            self.robot,
            guard,
        )

        # Step 2:
        # 0.5 -> 0.75
        run_policy_step(
            self.observation_source,
            policy,
            self.robot,
            guard,
        )

        # Step 3:
        # 0.75 -> 0.875
        run_policy_step(
            self.observation_source,
            policy,
            self.robot,
            guard,
        )

        np.testing.assert_allclose(
            self.robot.get_state().q,
            np.full(6, 0.875),
        )

        self.assertEqual(
            policy.predict_calls,
            3,
        )

    def test_policy_step_from_existing_observation(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.5,
        )

        guard = ActionGuard(
            ActionGuardConfig()
        )

        observation = (
            self.observation_source
            .get_observation()
        )

        result = run_policy_step_from_observation(
            observation=observation,
            policy=policy,
            robot=self.robot,
            action_guard=guard,
        )

        self.assertIs(
            result.observation,
            observation,
        )

        np.testing.assert_allclose(
            result.raw_action.q,
            np.full(6, 0.5),
        )

        np.testing.assert_allclose(
            self.robot.get_state().q,
            np.full(6, 0.5),
        )

if __name__ == "__main__":
    unittest.main()