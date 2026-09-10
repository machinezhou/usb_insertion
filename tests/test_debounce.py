import unittest

from usb_insertion.vision.debounce import (
    ConsecutiveTrue,
)


class TestConsecutiveTrue(
    unittest.TestCase
):
    def test_confirm_after_required_frames(self):
        debounce = ConsecutiveTrue(
            required=3
        )

        self.assertFalse(
            debounce.update(True)
        )

        self.assertFalse(
            debounce.update(True)
        )

        self.assertTrue(
            debounce.update(True)
        )

        self.assertEqual(
            debounce.count,
            3,
        )

    def test_false_resets_counter(self):
        debounce = ConsecutiveTrue(
            required=3
        )

        debounce.update(True)
        debounce.update(True)

        self.assertEqual(
            debounce.count,
            2,
        )

        result = debounce.update(
            False
        )

        self.assertFalse(
            result
        )

        self.assertEqual(
            debounce.count,
            0,
        )

    def test_can_confirm_again_after_reset(self):
        debounce = ConsecutiveTrue(
            required=2
        )

        debounce.update(True)
        self.assertTrue(
            debounce.update(True)
        )

        debounce.reset()

        self.assertEqual(
            debounce.count,
            0,
        )

        self.assertFalse(
            debounce.confirmed
        )

    def test_invalid_required_fails(self):
        with self.assertRaises(
            ValueError
        ):
            ConsecutiveTrue(
                required=0
            )


if __name__ == "__main__":
    unittest.main()