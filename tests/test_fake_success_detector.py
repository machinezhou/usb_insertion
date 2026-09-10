import unittest

from usb_insertion.core.datatypes import FrameBundle
from usb_insertion.fakes.fake_success_detector import (
    FakeSuccessDetector,
)


class TestFakeSuccessDetector(unittest.TestCase):
    def make_frames(self) -> FrameBundle:
        return FrameBundle(
            frames={},
            timestamp_s=1.0,
        )

    def test_success_after_three_updates(self):
        detector = FakeSuccessDetector(
            success_after_updates=3
        )

        frames = self.make_frames()

        result1 = detector.update(frames)
        result2 = detector.update(frames)
        result3 = detector.update(frames)

        self.assertFalse(result1.success)
        self.assertFalse(result2.success)
        self.assertTrue(result3.success)

    def test_update_count(self):
        detector = FakeSuccessDetector(
            success_after_updates=5
        )

        frames = self.make_frames()

        detector.update(frames)
        detector.update(frames)

        self.assertEqual(
            detector.update_count,
            2,
        )

    def test_reset_clears_update_count(self):
        detector = FakeSuccessDetector(
            success_after_updates=3
        )

        frames = self.make_frames()

        detector.update(frames)
        detector.update(frames)

        detector.reset()

        self.assertEqual(
            detector.update_count,
            0,
        )

        # 第一次 reset 只是开始 Attempt 1
        self.assertEqual(
            detector.attempt_index,
            0,
        )

    def test_attempt_schedule_fail_then_success(self):
        detector = FakeSuccessDetector(
            attempt_schedule=[
                None,
                2,
            ]
        )

        frames = self.make_frames()

        # Attempt 1
        detector.reset()

        self.assertEqual(
            detector.attempt_index,
            0,
        )

        for _ in range(5):
            result = detector.update(frames)

            self.assertFalse(
                result.success
            )

        # Attempt 2
        detector.reset()

        self.assertEqual(
            detector.attempt_index,
            1,
        )

        result1 = detector.update(frames)
        result2 = detector.update(frames)

        self.assertFalse(
            result1.success
        )

        self.assertTrue(
            result2.success
        )

    def test_schedule_exhausted_never_succeeds(self):
        detector = FakeSuccessDetector(
            attempt_schedule=[1]
        )

        frames = self.make_frames()

        # Attempt 1
        detector.reset()

        self.assertEqual(
            detector.attempt_index,
            0,
        )

        result = detector.update(frames)

        self.assertTrue(
            result.success
        )

        # Attempt 2：schedule 已耗尽
        detector.reset()

        self.assertEqual(
            detector.attempt_index,
            1,
        )

        for _ in range(5):
            result = detector.update(frames)

            self.assertFalse(
                result.success
            )

    def test_never_success(self):
        detector = FakeSuccessDetector(
            success_after_updates=None
        )

        frames = self.make_frames()

        for _ in range(10):
            result = detector.update(frames)

            self.assertFalse(
                result.success
            )

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            FakeSuccessDetector(
                success_after_updates=0
            )

    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            FakeSuccessDetector(
                attempt_schedule=[]
            )

        with self.assertRaises(ValueError):
            FakeSuccessDetector(
                attempt_schedule=[0]
            )


if __name__ == "__main__":
    unittest.main()