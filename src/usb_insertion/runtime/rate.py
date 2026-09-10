from __future__ import annotations

import time
from collections.abc import Callable


class RateLimiter:
    """
    基于绝对 deadline 的控制周期调度器。

    避免：
        process time + sleep(dt)

    产生累计漂移。
    """

    def __init__(
        self,
        fps: float,
        *,
        clock_fn: Callable[
            [],
            float,
        ] = time.monotonic,
        sleep_fn: Callable[
            [float],
            None,
        ] = time.sleep,
    ):
        if fps <= 0:
            raise ValueError(
                "fps must be > 0"
            )

        self._period_s = (
            1.0 / fps
        )

        self._clock_fn = (
            clock_fn
        )

        self._sleep_fn = (
            sleep_fn
        )

        self._next_deadline: (
            float | None
        ) = None

    def reset(self) -> None:
        self._next_deadline = (
            self._clock_fn()
        )

    def wait(self) -> None:
        if self._next_deadline is None:
            self.reset()

        assert (
            self._next_deadline
            is not None
        )

        self._next_deadline += (
            self._period_s
        )

        remaining = (
            self._next_deadline
            - self._clock_fn()
        )

        if remaining > 0:
            self._sleep_fn(
                remaining
            )
        else:
            # 已经落后一个或多个周期时，
            # 不连续补发历史 command。
            now = self._clock_fn()

            if (
                now
                - self._next_deadline
                > self._period_s
            ):
                self._next_deadline = now
