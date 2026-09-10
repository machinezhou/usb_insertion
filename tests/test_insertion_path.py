import tempfile
import unittest
from pathlib import Path

import numpy as np

from usb_insertion.vision.insertion.path import (
    InsertionPose,
    build_insertion_joint_delta,
    load_insertion_pose,
    save_insertion_joint_delta,
    save_insertion_pose,
)


class TestInsertionPath(
    unittest.TestCase
):
    def test_pose_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.json"

            pose = InsertionPose(
                stage="preinsert",
                q=np.arange(
                    6,
                    dtype=np.float64,
                ),
                timestamp_s=1.25,
            )

            save_insertion_pose(
                path,
                pose,
            )

            loaded = (
                load_insertion_pose(
                    path
                )
            )

            self.assertEqual(
                loaded.stage,
                "preinsert",
            )

            np.testing.assert_allclose(
                loaded.q,
                pose.q,
            )

    def test_gripper_zero(self):
        pre = np.zeros(6)

        inserted = np.array(
            [
                1,
                2,
                3,
                4,
                5,
                0.5,
            ],
            dtype=float,
        )

        delta = (
            build_insertion_joint_delta(
                pre,
                inserted,
                max_gripper_delta=1,
            )
        )

        self.assertEqual(
            delta[5],
            0.0,
        )

    def test_large_gripper_change_fails(
        self,
    ):
        pre = np.zeros(6)
        inserted = np.zeros(6)

        inserted[0] = 1
        inserted[5] = 2

        with self.assertRaises(
            ValueError
        ):
            build_insertion_joint_delta(
                pre,
                inserted,
                max_gripper_delta=0.5,
            )

    def test_save_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.npy"

            delta = np.array(
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                    0,
                ],
                dtype=float,
            )

            save_insertion_joint_delta(
                path,
                delta,
            )

            np.testing.assert_allclose(
                np.load(path),
                delta,
            )


if __name__ == "__main__":
    unittest.main()
