import unittest

import numpy as np

from usb_insertion.core.datatypes import (
    FrameBundle,
    Observation,
    RobotState,
)
from usb_insertion.fakes.fake_policy import (
    FakePolicy,
)


class TestFakePolicy(unittest.TestCase):
    def make_observation(
        self,
        q: np.ndarray,
    ) -> Observation:
        return Observation(
            robot=RobotState(
                q=q,
                timestamp_s=1.0,
            ),
            cameras=FrameBundle(
                frames={},
                timestamp_s=1.0,
            ),
        )

    def test_predict_moves_toward_target(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.25,
        )

        observation = self.make_observation(
            np.zeros(6)
        )

        action = policy.predict(
            observation
        )

        np.testing.assert_allclose(
            action.q,
            np.full(6, 0.25),
        )

    def test_gain_one_reaches_target(self):
        target_q = np.array([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ])

        policy = FakePolicy(
            target_q=target_q,
            gain=1.0,
        )

        observation = self.make_observation(
            np.zeros(6)
        )

        action = policy.predict(
            observation
        )

        np.testing.assert_allclose(
            action.q,
            target_q,
        )

    def test_multiple_predictions(self):
        policy = FakePolicy(
            target_q=np.ones(6),
            gain=0.5,
        )

        q = np.zeros(6)

        observation = self.make_observation(q)
        action1 = policy.predict(observation)

        observation = self.make_observation(
            action1.q
        )
        action2 = policy.predict(observation)

        np.testing.assert_allclose(
            action1.q,
            np.full(6, 0.5),
        )

        np.testing.assert_allclose(
            action2.q,
            np.full(6, 0.75),
        )

    def test_predict_counters(self):
        policy = FakePolicy(
            target_q=np.ones(6)
        )

        observation = self.make_observation(
            np.zeros(6)
        )

        policy.predict(observation)
        policy.predict(observation)

        self.assertEqual(
            policy.predict_calls,
            2,
        )

        self.assertEqual(
            policy.episode_step,
            2,
        )

    def test_reset_episode_step(self):
        policy = FakePolicy(
            target_q=np.ones(6)
        )

        observation = self.make_observation(
            np.zeros(6)
        )

        policy.predict(observation)
        policy.predict(observation)

        policy.reset()

        self.assertEqual(
            policy.episode_step,
            0,
        )

        self.assertEqual(
            policy.predict_calls,
            2,
        )

    def test_shape_mismatch_fails(self):
        policy = FakePolicy(
            target_q=np.ones(6)
        )

        observation = self.make_observation(
            np.zeros(5)
        )

        with self.assertRaises(
            ValueError
        ):
            policy.predict(
                observation
            )

    def test_invalid_gain_fails(self):
        with self.assertRaises(
            ValueError
        ):
            FakePolicy(
                target_q=np.ones(6),
                gain=0.0,
            )

        with self.assertRaises(
            ValueError
        ):
            FakePolicy(
                target_q=np.ones(6),
                gain=1.1,
            )

    def test_nan_target_fails(self):
        target_q = np.zeros(6)
        target_q[2] = np.nan

        with self.assertRaises(
            ValueError
        ):
            FakePolicy(
                target_q=target_q
            )


if __name__ == "__main__":
    unittest.main()