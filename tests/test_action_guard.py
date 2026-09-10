import unittest

import numpy as np

from usb_insertion.control.action_guard import (
    ActionGuard,
    ActionGuardConfig,
)
from usb_insertion.core.datatypes import Action


class TestActionGuard(unittest.TestCase):
    def test_pass_through(self):
        guard = ActionGuard(
            ActionGuardConfig()
        )

        current_q = np.zeros(6)
        target_q = np.ones(6)

        result = guard.apply(
            Action(q=target_q),
            current_q=current_q,
        )

        np.testing.assert_allclose(
            result.q,
            target_q,
        )

    def test_delta_limit(self):
        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=np.full(
                    6,
                    0.1,
                )
            )
        )

        current_q = np.zeros(6)
        target_q = np.ones(6)

        result = guard.apply(
            Action(q=target_q),
            current_q=current_q,
        )

        np.testing.assert_allclose(
            result.q,
            np.full(6, 0.1),
        )

    def test_delta_limit_negative_direction(self):
        guard = ActionGuard(
            ActionGuardConfig(
                max_delta_per_step=np.full(
                    6,
                    0.1,
                )
            )
        )

        current_q = np.zeros(6)
        target_q = -np.ones(6)

        result = guard.apply(
            Action(q=target_q),
            current_q=current_q,
        )

        np.testing.assert_allclose(
            result.q,
            np.full(6, -0.1),
        )

    def test_fixed_gripper(self):
        guard = ActionGuard(
            ActionGuardConfig(
                gripper_index=5,
                fixed_gripper_position=0.25,
            )
        )

        current_q = np.zeros(6)
        target_q = np.ones(6)

        result = guard.apply(
            Action(q=target_q),
            current_q=current_q,
        )

        self.assertAlmostEqual(
            result.q[5],
            0.25,
        )

    def test_joint_limits(self):
        guard = ActionGuard(
            ActionGuardConfig(
                joint_min=np.full(6, -1.0),
                joint_max=np.full(6, 1.0),
            )
        )

        current_q = np.zeros(6)

        target_q = np.array([
            -2.0,
            -0.5,
            0.0,
            0.5,
            2.0,
            0.0,
        ])

        result = guard.apply(
            Action(q=target_q),
            current_q=current_q,
        )

        expected = np.array([
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            0.0,
        ])

        np.testing.assert_allclose(
            result.q,
            expected,
        )

    def test_nan_rejected(self):
        guard = ActionGuard(
            ActionGuardConfig()
        )

        target_q = np.zeros(6)
        target_q[2] = np.nan

        with self.assertRaises(ValueError):
            guard.apply(
                Action(q=target_q),
                current_q=np.zeros(6),
            )

    def test_inf_rejected(self):
        guard = ActionGuard(
            ActionGuardConfig()
        )

        target_q = np.zeros(6)
        target_q[3] = np.inf

        with self.assertRaises(ValueError):
            guard.apply(
                Action(q=target_q),
                current_q=np.zeros(6),
            )

    def test_shape_mismatch_rejected(self):
        guard = ActionGuard(
            ActionGuardConfig()
        )

        with self.assertRaises(ValueError):
            guard.apply(
                Action(q=np.zeros(6)),
                current_q=np.zeros(5),
            )

    def test_invalid_gripper_index(self):
        guard = ActionGuard(
            ActionGuardConfig(
                gripper_index=10,
                fixed_gripper_position=0.25,
            )
        )

        with self.assertRaises(ValueError):
            guard.apply(
                Action(q=np.zeros(6)),
                current_q=np.zeros(6),
            )


if __name__ == "__main__":
    unittest.main()