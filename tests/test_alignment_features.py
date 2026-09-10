import unittest

from usb_insertion.vision.alignment.features import (
    angle_between_points_deg,
    wrap_angle_deg,
)


class TestAlignmentFeatures(
    unittest.TestCase
):
    def test_wrap_angle(self):
        self.assertAlmostEqual(
            wrap_angle_deg(190),
            -170,
        )

        self.assertAlmostEqual(
            wrap_angle_deg(-190),
            170,
        )

    def test_angle_horizontal(self):
        angle = (
            angle_between_points_deg(
                (0, 0),
                (10, 0),
            )
        )

        self.assertAlmostEqual(
            angle,
            0.0,
        )

    def test_angle_vertical(self):
        angle = (
            angle_between_points_deg(
                (0, 0),
                (0, 10),
            )
        )

        self.assertAlmostEqual(
            angle,
            90.0,
        )


if __name__ == "__main__":
    unittest.main()