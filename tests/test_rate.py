import unittest

from usb_insertion.runtime.rate import (
    RateLimiter,
)


class FakeTime:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def clock(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(
            seconds
        )
        self.value += seconds


class TestRateLimiter(
    unittest.TestCase
):
    def test_wait(self):
        fake = FakeTime()

        rate = RateLimiter(
            10,
            clock_fn=fake.clock,
            sleep_fn=fake.sleep,
        )

        rate.reset()
        rate.wait()

        self.assertAlmostEqual(
            fake.value,
            0.1,
        )

        rate.wait()

        self.assertAlmostEqual(
            fake.value,
            0.2,
        )

    def test_invalid_fps(self):
        with self.assertRaises(
            ValueError
        ):
            RateLimiter(0)


if __name__ == "__main__":
    unittest.main()
