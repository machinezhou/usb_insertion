import unittest

import numpy as np

from usb_insertion.control.trajectory_planner import (
    QUINTIC_MAX_ACCELERATION_FACTOR,
    TrajectoryLimits,
    minimum_duration,
    plan_joint_trajectory,
)


class TestTrajectoryPlanner(
    unittest.TestCase
):
    def test_requested_duration_is_respected(self):
        duration = minimum_duration(
            q_start=np.zeros(6),
            q_target=np.ones(6),
            fps=30,
            limits=TrajectoryLimits(),
            requested_duration_s=3.0,
        )

        self.assertEqual(
            duration,
            3.0,
        )

    def test_velocity_limit(self):
        q_start = np.zeros(1)
        q_target = np.ones(1)

        duration = minimum_duration(
            q_start=q_start,
            q_target=q_target,
            fps=8,
            limits=TrajectoryLimits(
                max_velocity=np.array([
                    1.0
                ])
            ),
        )

        # quintic max velocity factor:
        #
        # 1.875 * delta / vmax
        #
        # delta = 1
        # vmax = 1
        #
        # duration = 1.875 s
        self.assertAlmostEqual(
            duration,
            1.875,
        )

    def test_acceleration_limit(self):
        q_start = np.zeros(1)
        q_target = np.ones(1)

        duration = minimum_duration(
            q_start=q_start,
            q_target=q_target,
            fps=100,
            limits=TrajectoryLimits(
                max_acceleration=np.array([
                    QUINTIC_MAX_ACCELERATION_FACTOR
                ])
            ),
        )

        # delta = 1
        #
        # max_acceleration 设置为
        # quintic acceleration factor，
        #
        # 因此理论最短时间正好是 1 秒。
        self.assertAlmostEqual(
            duration,
            1.0,
        )

    def test_larger_constraint_wins(self):
        duration = minimum_duration(
            q_start=np.zeros(1),
            q_target=np.ones(1),
            fps=10,
            limits=TrajectoryLimits(
                max_velocity=np.array([
                    10.0
                ])
            ),
            requested_duration_s=3.0,
        )

        self.assertEqual(
            duration,
            3.0,
        )

    def test_plan_endpoints(self):
        q_start = np.zeros(6)

        q_target = np.array([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ])

        plan = plan_joint_trajectory(
            q_start=q_start,
            q_target=q_target,
            fps=30,
            limits=TrajectoryLimits(
                max_velocity=np.full(
                    6,
                    10.0,
                )
            ),
        )

        np.testing.assert_allclose(
            plan.trajectory[0],
            q_start,
        )

        np.testing.assert_allclose(
            plan.trajectory[-1],
            q_target,
        )

    def test_per_step_delta_is_respected(self):
        q_start = np.zeros(6)
        q_target = np.ones(6)

        max_delta = np.full(
            6,
            0.05,
        )

        plan = plan_joint_trajectory(
            q_start=q_start,
            q_target=q_target,
            fps=30,
            limits=TrajectoryLimits(
                max_delta_per_step=max_delta
            ),
        )

        actual_delta = np.abs(
            np.diff(
                plan.trajectory,
                axis=0,
            )
        )

        self.assertTrue(
            np.all(
                actual_delta
                <= max_delta[None, :]
                + 1e-12
            )
        )

    def test_stationary_motion(self):
        q = np.ones(6)

        plan = plan_joint_trajectory(
            q_start=q,
            q_target=q,
            fps=30,
            limits=TrajectoryLimits(),
        )

        np.testing.assert_allclose(
            plan.trajectory[0],
            q,
        )

        np.testing.assert_allclose(
            plan.trajectory[-1],
            q,
        )

        self.assertGreater(
            plan.duration_s,
            0,
        )

    def test_invalid_velocity_limit(self):
        with self.assertRaises(
            ValueError
        ):
            minimum_duration(
                q_start=np.zeros(6),
                q_target=np.ones(6),
                fps=30,
                limits=TrajectoryLimits(
                    max_velocity=np.zeros(6)
                ),
            )

    def test_limit_shape_mismatch(self):
        with self.assertRaises(
            ValueError
        ):
            minimum_duration(
                q_start=np.zeros(6),
                q_target=np.ones(6),
                fps=30,
                limits=TrajectoryLimits(
                    max_velocity=np.ones(5)
                ),
            )


if __name__ == "__main__":
    unittest.main()