import unittest

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
)


class TestAlignmentEstimate(
    unittest.TestCase
):
    def test_alignment_estimate(self):
        result = AlignmentEstimate(
            visible=True,
            in_capture_region=True,
            aligned=False,
            errors={
                "top_dx_px": 12.5,
            },
            confidence=0.95,
            plug_face="marked",
            reason="detected",
        )

        self.assertTrue(
            result.visible
        )

        self.assertTrue(
            result.in_capture_region
        )

        self.assertFalse(
            result.aligned
        )

        self.assertEqual(
            result.plug_face,
            "marked",
        )

    def test_defaults(self):
        result = AlignmentEstimate(
            visible=False,
            in_capture_region=False,
            aligned=False,
        )

        self.assertEqual(
            dict(result.errors),
            {},
        )


if __name__ == "__main__":
    unittest.main()