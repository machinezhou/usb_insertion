import unittest

import numpy as np

from usb_insertion.control.trajectory import (
    joint_trajectory,
    quintic_smoothstep,
)


class TestTrajectory(unittest.TestCase):
    def test_smoothstep_endpoints(self):
        u = np.array([0.0, 1.0])

        s = quintic_smoothstep(u)

        self.assertAlmostEqual(s[0], 0.0)
        self.assertAlmostEqual(s[1], 1.0)

    def test_trajectory_endpoints(self):
        q_start = np.array([0.0, 1.0, 2.0])
        q_target = np.array([1.0, 2.0, 3.0])

        trajectory = joint_trajectory(
            q_start=q_start,
            q_target=q_target,
            duration_s=1.0,
            fps=30,
        )

        np.testing.assert_allclose(
            trajectory[0],
            q_start,
        )

        np.testing.assert_allclose(
            trajectory[-1],
            q_target,
        )

    def test_trajectory_shape(self):
        trajectory = joint_trajectory(
            q_start=np.zeros(6),
            q_target=np.ones(6),
            duration_s=2.0,
            fps=30,
        )

        self.assertEqual(
            trajectory.shape,
            (61, 6),
        )

    def test_stationary_trajectory(self):
        q = np.array([1.0, 2.0, 3.0])

        trajectory = joint_trajectory(
            q_start=q,
            q_target=q,
            duration_s=1.0,
            fps=30,
        )

        expected = np.repeat(
            q[None, :],
            31,
            axis=0,
        )

        np.testing.assert_allclose(
            trajectory,
            expected,
        )

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            joint_trajectory(
                q_start=np.zeros(6),
                q_target=np.zeros(5),
                duration_s=1.0,
                fps=30,
            )

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            joint_trajectory(
                q_start=np.zeros(6),
                q_target=np.ones(6),
                duration_s=0.0,
                fps=30,
            )


if __name__ == "__main__":
    unittest.main()