import unittest

import numpy as np

from usb_insertion.core.datatypes import Action
from usb_insertion.fakes.fake_robot import FakeRobot


class TestFakeRobot(unittest.TestCase):
    def test_initial_state(self):
        robot = FakeRobot(dof=6)

        robot.connect()

        state = robot.get_state()

        np.testing.assert_allclose(
            state.q,
            np.zeros(6),
        )

        robot.close()

    def test_custom_initial_state(self):
        initial_q = np.array([
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ])

        robot = FakeRobot(
            dof=6,
            initial_q=initial_q,
        )

        robot.connect()

        state = robot.get_state()

        np.testing.assert_allclose(
            state.q,
            initial_q,
        )

    def test_send_action_updates_state(self):
        robot = FakeRobot(dof=6)
        robot.connect()

        target_q = np.array([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ])

        robot.send_action(
            Action(q=target_q)
        )

        state = robot.get_state()

        np.testing.assert_allclose(
            state.q,
            target_q,
        )

    def test_state_returns_copy(self):
        robot = FakeRobot(dof=6)
        robot.connect()

        state = robot.get_state()

        state.q[0] = 100.0

        new_state = robot.get_state()

        self.assertEqual(
            new_state.q[0],
            0.0,
        )

    def test_send_action_before_connect_fails(self):
        robot = FakeRobot(dof=6)

        with self.assertRaises(RuntimeError):
            robot.send_action(
                Action(q=np.zeros(6))
            )

    def test_get_state_before_connect_fails(self):
        robot = FakeRobot(dof=6)

        with self.assertRaises(RuntimeError):
            robot.get_state()

    def test_wrong_action_shape_fails(self):
        robot = FakeRobot(dof=6)
        robot.connect()

        with self.assertRaises(ValueError):
            robot.send_action(
                Action(q=np.zeros(5))
            )

    def test_nan_action_fails(self):
        robot = FakeRobot(dof=6)
        robot.connect()

        q = np.zeros(6)
        q[2] = np.nan

        with self.assertRaises(ValueError):
            robot.send_action(
                Action(q=q)
            )

    def test_close_disconnects_robot(self):
        robot = FakeRobot(dof=6)

        robot.connect()
        self.assertTrue(robot.connected)

        robot.close()
        self.assertFalse(robot.connected)


if __name__ == "__main__":
    unittest.main()