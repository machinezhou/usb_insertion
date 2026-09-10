import unittest

import numpy as np

from usb_insertion.control.action_guard import (
    ActionGuard,
    ActionGuardConfig,
)
from usb_insertion.control.scripted_motion import (
    MotionResult,
    ScriptedMotionController,
)
from usb_insertion.control.trajectory_planner import (
    TrajectoryLimits,
)
from usb_insertion.fakes.fake_robot import (
    FakeRobot,
)


class TestScriptedMotionController(
    unittest.TestCase
):
    def setUp(self):
        self.robot = FakeRobot(
            dof=6
        )

        self.robot.connect()

    def tearDown(self):
        self.robot.close()

    def test_move_to_target(self):
        guard = ActionGuard(
            ActionGuardConfig()
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=30,
        )

        target_q = np.array([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ])

        result = controller.move_to(
            target_q=target_q,
            duration_s=1.0,
            realtime=False,
        )

        self.assertIsInstance(
            result,
            MotionResult,
        )

        np.testing.assert_allclose(
            self.robot.get_state().q,
            target_q,
        )

        np.testing.assert_allclose(
            result.final_command_q,
            target_q,
        )

        np.testing.assert_allclose(
            result.target_q,
            target_q,
        )

    def test_command_step_count(self):
        guard = ActionGuard(
            ActionGuardConfig()
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=30,
        )

        result = controller.move_to(
            target_q=np.ones(6),
            duration_s=2.0,
            realtime=False,
        )

        # 2 秒 @ 30 Hz：
        #
        # trajectory 总共有：
        # 2 * 30 + 1 = 61 个点
        #
        # 第一个点是当前状态，
        # 所以真正发送 60 个 command。
        self.assertEqual(
            result.command_steps,
            60,
        )

        self.assertEqual(
            result.planned_duration_s,
            2.0,
        )

    def test_fixed_gripper(self):
        guard = ActionGuard(
            ActionGuardConfig(
                gripper_index=5,
                fixed_gripper_position=0.25,
            )
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=30,
        )

        target_q = np.ones(6)

        result = controller.move_to(
            target_q=target_q,
            duration_s=1.0,
            realtime=False,
        )

        self.assertAlmostEqual(
            result.final_command_q[5],
            0.25,
        )

        self.assertAlmostEqual(
            self.robot.get_state().q[5],
            0.25,
        )

        # 其他 5 个关节仍然应该到达目标。
        np.testing.assert_allclose(
            result.final_command_q[:5],
            np.ones(5),
        )

    def test_planner_respects_action_guard_delta(self):
        """
        如果 ActionGuard 规定很严格的单步最大变化，
        Planner 应该主动延长 duration。

        而不是先生成过快轨迹，
        再被 ActionGuard 一路截断，
        导致最后到不了目标。
        """

        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=np.full(
                    6,
                    0.01,
                )
            )
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=2,
        )

        target_q = np.ones(6)

        result = controller.move_to(
            target_q=target_q,
            duration_s=1.0,
            realtime=False,
        )

        # 用户只请求 1 秒，
        # 但 0 -> 1 且每步最多 0.01，
        # Planner 必须自动把时间拉长。
        self.assertGreater(
            result.planned_duration_s,
            1.0,
        )

        # 最终应该真正到达目标，
        # 而不是像旧实现那样只走到 0.02。
        np.testing.assert_allclose(
            result.final_command_q,
            target_q,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            self.robot.get_state().q,
            target_q,
            atol=1e-12,
        )

    def test_explicit_trajectory_limits_are_used(self):
        """
        验证显式传入的 Planner 速度限制会生效，
        并且最终 duration 会向上对齐到完整控制周期。
        """

        guard = ActionGuard(
            ActionGuardConfig()
        )

        limits = TrajectoryLimits(
            max_velocity=np.full(
                6,
                0.5,
            )
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=30,
            trajectory_limits=limits,
        )

        result = controller.move_to(
            target_q=np.ones(6),
            duration_s=1.0,
            realtime=False,
        )

        # 五次 smoothstep 峰值速度系数：
        #
        # 1.875
        #
        # delta = 1
        # vmax = 0.5
        #
        # 连续时间理论最短：
        #
        # 1.875 / 0.5 = 3.75 秒
        #
        # 但：
        #
        # 3.75 * 30 = 112.5 个控制周期
        #
        # 必须向上取整到 113 个周期：
        #
        # 113 / 30 = 3.766666... 秒

        expected_duration = 113 / 30

        self.assertAlmostEqual(
            result.planned_duration_s,
            expected_duration,
        )

        np.testing.assert_allclose(
            result.final_command_q,
            np.ones(6),
        )

    def test_planner_uses_stricter_delta_limit(self):
        """
        Planner 和 ActionGuard 都有 max_delta 时，
        ScriptedMotionController 应使用更严格的一组。
        """

        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=np.full(
                    6,
                    0.02,
                )
            )
        )

        limits = TrajectoryLimits(
            max_delta_per_step=np.full(
                6,
                0.05,
            )
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=30,
            trajectory_limits=limits,
        )

        effective_delta = (
            controller
            .trajectory_limits
            .max_delta_per_step
        )

        np.testing.assert_allclose(
            effective_delta,
            np.full(
                6,
                0.02,
            ),
        )

    def test_planner_inherits_guard_delta_limit(self):
        """
        如果 Planner 自己没有 max_delta，
        但 ActionGuard 有，
        Planner 应自动继承 ActionGuard 的限制。
        """

        guard_delta = np.full(
            6,
            0.03,
        )

        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=guard_delta
            )
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=30,
            trajectory_limits=TrajectoryLimits(),
        )

        effective_delta = (
            controller
            .trajectory_limits
            .max_delta_per_step
        )

        np.testing.assert_allclose(
            effective_delta,
            guard_delta,
        )

    def test_start_q_is_recorded(self):
        initial_q = np.array([
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ])

        self.robot.close()

        self.robot = FakeRobot(
            dof=6,
            initial_q=initial_q,
        )

        self.robot.connect()

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=ActionGuard(
                ActionGuardConfig()
            ),
            control_fps=30,
        )

        result = controller.move_to(
            target_q=np.ones(6),
            duration_s=1.0,
            realtime=False,
        )

        np.testing.assert_allclose(
            result.start_q,
            initial_q,
        )

    def test_duration_can_be_automatic(self):
        """
        duration_s=None 时，
        Planner 可以完全根据约束自动决定运动时间。
        """

        guard = ActionGuard(
            ActionGuardConfig()
        )

        limits = TrajectoryLimits(
            max_velocity=np.full(
                6,
                1.0,
            )
        )

        controller = ScriptedMotionController(
            robot=self.robot,
            action_guard=guard,
            control_fps=30,
            trajectory_limits=limits,
        )

        result = controller.move_to(
            target_q=np.ones(6),
            duration_s=None,
            realtime=False,
        )

        self.assertGreater(
            result.planned_duration_s,
            0.0,
        )

        np.testing.assert_allclose(
            result.final_command_q,
            np.ones(6),
        )

    def test_invalid_control_fps(self):
        with self.assertRaises(
            ValueError
        ):
            ScriptedMotionController(
                robot=self.robot,
                action_guard=ActionGuard(
                    ActionGuardConfig()
                ),
                control_fps=0,
            )

    def test_planner_guard_delta_shape_mismatch(self):
        """
        Planner 和 ActionGuard 的 max_delta 维度不一致，
        应在 Controller 初始化时直接拒绝。
        """

        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=np.ones(6)
            )
        )

        limits = TrajectoryLimits(
            max_delta_per_step=np.ones(5)
        )

        with self.assertRaises(
            ValueError
        ):
            ScriptedMotionController(
                robot=self.robot,
                action_guard=guard,
                control_fps=30,
                trajectory_limits=limits,
            )


if __name__ == "__main__":
    unittest.main()