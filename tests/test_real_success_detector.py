import unittest

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
    FrameBundle,
)
from usb_insertion.vision.insertion.success_detector import (
    AlignmentSuccessDetector,
    SuccessReference,
)


class TestAlignmentSuccessDetector(
    unittest.TestCase
):
    def test_consecutive_success(self):
        detector = (
            AlignmentSuccessDetector(
                SuccessReference(
                    target_errors={
                        "depth": 20.0,
                    },
                    tolerances={
                        "depth": 2.0,
                    },
                    required_consecutive=3,
                )
            )
        )

        frames = FrameBundle(
            frames={},
            timestamp_s=1.0,
        )

        alignment = (
            AlignmentEstimate(
                visible=True,
                in_capture_region=True,
                aligned=False,
                errors={
                    "depth": 20.5,
                },
            )
        )

        self.assertFalse(
            detector.update(
                frames,
                alignment,
            ).success
        )

        self.assertFalse(
            detector.update(
                frames,
                alignment,
            ).success
        )

        self.assertTrue(
            detector.update(
                frames,
                alignment,
            ).success
        )

    def test_bad_frame_resets(self):
        detector = (
            AlignmentSuccessDetector(
                SuccessReference(
                    target_errors={
                        "depth": 20,
                    },
                    tolerances={
                        "depth": 1,
                    },
                    required_consecutive=2,
                )
            )
        )

        frames = FrameBundle(
            frames={},
            timestamp_s=1,
        )

        good = AlignmentEstimate(
            visible=True,
            in_capture_region=True,
            aligned=False,
            errors={
                "depth": 20,
            },
        )

        bad = AlignmentEstimate(
            visible=True,
            in_capture_region=True,
            aligned=False,
            errors={
                "depth": 30,
            },
        )

        detector.update(
            frames,
            good,
        )

        result = detector.update(
            frames,
            bad,
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.consecutive,
            0,
        )


if __name__ == "__main__":
    unittest.main()
