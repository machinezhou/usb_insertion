import unittest

import numpy as np

from usb_insertion.fakes.fake_cameras import (
    FakeCameras,
)


class TestFakeCameras(unittest.TestCase):
    def test_default_camera_names(self):
        cameras = FakeCameras()

        self.assertEqual(
            cameras.camera_names,
            (
                "top",
                "wrist",
                "side",
            ),
        )

    def test_open_and_close(self):
        cameras = FakeCameras()

        self.assertFalse(
            cameras.opened
        )

        cameras.open()

        self.assertTrue(
            cameras.opened
        )

        cameras.close()

        self.assertFalse(
            cameras.opened
        )

    def test_read_before_open_fails(self):
        cameras = FakeCameras()

        with self.assertRaises(
            RuntimeError
        ):
            cameras.read()

    def test_read_returns_three_frames(self):
        cameras = FakeCameras()
        cameras.open()

        bundle = cameras.read()

        self.assertEqual(
            set(bundle.frames.keys()),
            {
                "top",
                "wrist",
                "side",
            },
        )

    def test_default_frame_shape(self):
        cameras = FakeCameras()
        cameras.open()

        bundle = cameras.read()

        for frame in bundle.frames.values():
            self.assertEqual(
                frame.shape,
                (480, 640, 3),
            )

    def test_default_frame_dtype(self):
        cameras = FakeCameras()
        cameras.open()

        bundle = cameras.read()

        for frame in bundle.frames.values():
            self.assertEqual(
                frame.dtype,
                np.uint8,
            )

    def test_custom_configuration(self):
        cameras = FakeCameras(
            camera_names=(
                "wrist",
                "side",
            ),
            width=320,
            height=240,
            channels=3,
        )

        cameras.open()

        bundle = cameras.read()

        self.assertEqual(
            set(bundle.frames.keys()),
            {
                "wrist",
                "side",
            },
        )

        self.assertEqual(
            bundle.frames["wrist"].shape,
            (240, 320, 3),
        )

    def test_duplicate_camera_names_fail(self):
        with self.assertRaises(
            ValueError
        ):
            FakeCameras(
                camera_names=(
                    "wrist",
                    "wrist",
                )
            )


if __name__ == "__main__":
    unittest.main()