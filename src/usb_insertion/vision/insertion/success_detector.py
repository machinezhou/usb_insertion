from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
    DetectionResult,
    FrameBundle,
)
from usb_insertion.vision.debounce import (
    ConsecutiveTrue,
)


@dataclass(frozen=True, slots=True)
class SuccessReference:
    """
    完全插到底时对应的视觉误差参考。

    注意：
    AlignmentEstimator 的 errors
    是相对 PREINSERT reference 的。

    因此 USB 完全插入后，
    errors 通常不是 0。

    我们后续实际插到底一次，
    自动采集这个 target_errors。
    """

    target_errors: dict[str, float]

    tolerances: dict[str, float]

    required_consecutive: int


def load_success_reference(
    path: str | Path,
) -> SuccessReference:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(
            file
        )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            "success reference root "
            "must be mapping"
        )

    target_raw = raw.get(
        "target_errors"
    )

    tolerance_raw = raw.get(
        "tolerances"
    )

    if not isinstance(
        target_raw,
        dict,
    ):
        raise ValueError(
            "target_errors must "
            "be mapping"
        )

    if not isinstance(
        tolerance_raw,
        dict,
    ):
        raise ValueError(
            "tolerances must "
            "be mapping"
        )

    targets = {
        str(key): float(value)
        for key, value
        in target_raw.items()
    }

    tolerances = {
        str(key): float(value)
        for key, value
        in tolerance_raw.items()
    }

    if set(targets) != set(
        tolerances
    ):
        raise ValueError(
            "target_errors and "
            "tolerances keys mismatch"
        )

    for key, value in (
        tolerances.items()
    ):
        if value <= 0:
            raise ValueError(
                f"{key} tolerance "
                "must be > 0"
            )

    required = int(
        raw.get(
            "required_consecutive",
            12,
        )
    )

    return SuccessReference(
        target_errors=targets,
        tolerances=tolerances,
        required_consecutive=required,
    )


class AlignmentSuccessDetector:
    """
    几何插入成功检测器。

    candidate success:
        当前视觉 error
        接近完全插入 reference

    final success:
        candidate 连续 N 帧成立
    """

    def __init__(
        self,
        reference: SuccessReference,
    ):
        self._reference = reference

        self._debounce = (
            ConsecutiveTrue(
                reference
                .required_consecutive
            )
        )

    def reset(self) -> None:
        self._debounce.reset()

    def update(
        self,
        frames: FrameBundle,
        alignment: AlignmentEstimate | None = None,
    ) -> DetectionResult:
        del frames

        if (
            alignment is None
            or not alignment.visible
        ):
            self._debounce.update(
                False
            )

            return DetectionResult(
                success=False,
                score=0.0,
                consecutive=0,
                reason=(
                    "alignment_unavailable"
                ),
            )

        worst_ratio = 0.0

        for key, target in (
            self._reference
            .target_errors
            .items()
        ):
            if key not in (
                alignment.errors
            ):
                self._debounce.update(
                    False
                )

                return DetectionResult(
                    success=False,
                    score=0.0,
                    consecutive=0,
                    reason=(
                        "missing_error:"
                        f"{key}"
                    ),
                )

            current = float(
                alignment.errors[
                    key
                ]
            )

            tolerance = (
                self._reference
                .tolerances[
                    key
                ]
            )

            ratio = (
                abs(
                    current
                    - target
                )
                / tolerance
            )

            worst_ratio = max(
                worst_ratio,
                ratio,
            )

        candidate = (
            worst_ratio <= 1.0
        )

        confirmed = (
            self._debounce.update(
                candidate
            )
        )

        score = max(
            0.0,
            min(
                1.0,
                1.0 - worst_ratio,
            ),
        )

        return DetectionResult(
            success=confirmed,
            score=score,
            consecutive=(
                self._debounce.count
            ),
            reason=(
                "inserted_confirmed"
                if confirmed
                else (
                    "inserted_candidate"
                    if candidate
                    else "not_inserted"
                )
            ),
        )
