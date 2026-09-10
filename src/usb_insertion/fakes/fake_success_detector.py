from __future__ import annotations

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
    DetectionResult,
    FrameBundle,
)


class FakeSuccessDetector:
    def __init__(
        self,
        success_after_updates: int | None = None,
        attempt_schedule: list[
            int | None
        ] | None = None,
    ):
        if (
            success_after_updates
            is not None
            and success_after_updates <= 0
        ):
            raise ValueError(
                "success_after_updates "
                "must be > 0 or None"
            )

        if attempt_schedule is not None:
            if not attempt_schedule:
                raise ValueError(
                    "attempt_schedule "
                    "must not be empty"
                )

            for value in (
                attempt_schedule
            ):
                if (
                    value is not None
                    and value <= 0
                ):
                    raise ValueError(
                        "schedule values "
                        "must be > 0 or None"
                    )

        self._success_after_updates = (
            success_after_updates
        )

        self._attempt_schedule = (
            attempt_schedule
        )

        self._attempt_index = 0
        self._update_count = 0
        self._started = False

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def attempt_index(self) -> int:
        return self._attempt_index

    def reset(self) -> None:
        if self._started:
            self._attempt_index += 1
        else:
            self._started = True

        self._update_count = 0

    def _threshold(
        self,
    ) -> int | None:
        if (
            self._attempt_schedule
            is None
        ):
            return (
                self
                ._success_after_updates
            )

        if self._attempt_index >= len(
            self._attempt_schedule
        ):
            return None

        return self._attempt_schedule[
            self._attempt_index
        ]

    def update(
        self,
        frames: FrameBundle,
        alignment: AlignmentEstimate | None = None,
    ) -> DetectionResult:
        del frames
        del alignment

        self._update_count += 1

        threshold = (
            self._threshold()
        )

        success = (
            threshold is not None
            and self._update_count
            >= threshold
        )

        return DetectionResult(
            success=success,
            score=(
                1.0
                if success
                else 0.0
            ),
            consecutive=(
                self._update_count
                if success
                else 0
            ),
            reason=(
                "fake_success"
                if success
                else "fake_pending"
            ),
        )
