import tempfile
import unittest
from pathlib import Path

import numpy as np

from usb_insertion.vision.alignment.jacobian import (
    load_image_jacobian,
    save_image_jacobian,
)


class TestImageJacobian(
    unittest.TestCase
):
    def test_load_npy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "jacobian.npy"
            )

            matrix = np.arange(
                15,
                dtype=np.float64,
            ).reshape(
                3,
                5,
            )

            np.save(
                path,
                matrix,
                allow_pickle=False,
            )

            model = (
                load_image_jacobian(
                    path,
                    error_keys=(
                        "e0",
                        "e1",
                        "e2",
                    ),
                    controlled_joint_indices=(
                        0,
                        1,
                        2,
                        3,
                        4,
                    ),
                )
            )

            np.testing.assert_allclose(
                model.matrix,
                matrix,
            )

            self.assertEqual(
                model.shape,
                (3, 5),
            )

    def test_save_and_load_npy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "jacobian.npy"
            )

            matrix = np.eye(
                3,
                dtype=np.float64,
            )

            save_image_jacobian(
                path,
                matrix,
            )

            model = (
                load_image_jacobian(
                    path,
                    error_keys=(
                        "e0",
                        "e1",
                        "e2",
                    ),
                    controlled_joint_indices=(
                        0,
                        1,
                        2,
                    ),
                )
            )

            np.testing.assert_allclose(
                model.matrix,
                matrix,
            )

    def test_load_legacy_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "jacobian.npz"
            )

            matrix = np.eye(
                2,
                dtype=np.float64,
            )

            np.savez(
                path,
                jacobian=matrix,
            )

            model = (
                load_image_jacobian(
                    path,
                    error_keys=(
                        "e0",
                        "e1",
                    ),
                    controlled_joint_indices=(
                        0,
                        1,
                    ),
                )
            )

            np.testing.assert_allclose(
                model.matrix,
                matrix,
            )

    def test_row_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "jacobian.npy"
            )

            np.save(
                path,
                np.eye(3),
            )

            with self.assertRaises(
                ValueError
            ):
                load_image_jacobian(
                    path,
                    error_keys=(
                        "e0",
                        "e1",
                    ),
                    controlled_joint_indices=(
                        0,
                        1,
                        2,
                    ),
                )

    def test_nan_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "jacobian.npy"
            )

            matrix = np.eye(
                2
            )

            matrix[
                0,
                0,
            ] = np.nan

            np.save(
                path,
                matrix,
            )

            with self.assertRaises(
                ValueError
            ):
                load_image_jacobian(
                    path
                )


if __name__ == "__main__":
    unittest.main()
