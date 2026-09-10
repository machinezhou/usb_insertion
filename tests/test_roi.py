import unittest

import numpy as np

from usb_insertion.vision.roi import (
    ImageROI,
)


class TestImageROI(unittest.TestCase):
    def test_properties(self):
        roi = ImageROI(
            x=10,
            y=20,
            width=100,
            height=50,
        )

        self.assertEqual(
            roi.x2,
            110,
        )

        self.assertEqual(
            roi.y2,
            70,
        )

    def test_crop_color_image(self):
        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        roi = ImageROI(
            x=100,
            y=50,
            width=200,
            height=120,
        )

        cropped = roi.crop(
            frame
        )

        self.assertEqual(
            cropped.shape,
            (120, 200, 3),
        )

    def test_crop_gray_image(self):
        frame = np.zeros(
            (480, 640),
            dtype=np.uint8,
        )

        roi = ImageROI(
            x=100,
            y=50,
            width=200,
            height=120,
        )

        cropped = roi.crop(
            frame
        )

        self.assertEqual(
            cropped.shape,
            (120, 200),
        )

    def test_copy_does_not_modify_original(self):
        frame = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        roi = ImageROI(
            x=10,
            y=10,
            width=20,
            height=20,
        )

        cropped = roi.crop(
            frame,
            copy=True,
        )

        cropped[:] = 255

        self.assertEqual(
            int(frame.sum()),
            0,
        )

    def test_out_of_bounds_fails(self):
        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        )

        roi = ImageROI(
            x=600,
            y=100,
            width=100,
            height=100,
        )

        with self.assertRaises(
            ValueError
        ):
            roi.crop(
                frame
            )

    def test_invalid_roi_fails(self):
        with self.assertRaises(
            ValueError
        ):
            ImageROI(
                x=-1,
                y=0,
                width=100,
                height=100,
            )

        with self.assertRaises(
            ValueError
        ):
            ImageROI(
                x=0,
                y=0,
                width=0,
                height=100,
            )


if __name__ == "__main__":
    unittest.main()