from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(
    frozen=True,
    slots=True,
)
class ACTRunConfig:
    checkpoint: str
    dataset_repo_id: str
    dataset_root: Path | None
    task: str
    device: str
    timeout_s: float


@dataclass(
    frozen=True,
    slots=True,
)
class InsertionRunConfig:
    joint_delta_path: Path
    success_reference_path: Path

    progress_step: float
    timeout_s: float

    correction_error_keys: tuple[
        str,
        ...,
    ]

    correction_gain: float
    correction_damping: float

    correction_max_joint_delta: tuple[
        float,
        ...,
    ]

    guard_error_limits: dict[
        str,
        float,
    ]

    retract_scale: float
    retract_duration_s: float


@dataclass(
    frozen=True,
    slots=True,
)
class RunConfig:
    control_fps: float

    home_q: tuple[
        float,
        ...,
    ]

    home_duration_s: float
    max_attempts: int

    visual_alignment_timeout_s: float

    max_delta_per_step: tuple[
        float,
        ...,
    ]

    act: ACTRunConfig
    insertion: InsertionRunConfig


def _positive_float(
    value: object,
    name: str,
) -> float:

    result = float(
        value
    )

    if (
        not math.isfinite(
            result
        )
        or result <= 0
    ):
        raise ValueError(
            f"{name} must be "
            "finite and > 0"
        )

    return result


def _nonnegative_float(
    value: object,
    name: str,
) -> float:

    result = float(
        value
    )

    if (
        not math.isfinite(
            result
        )
        or result < 0
    ):
        raise ValueError(
            f"{name} must be "
            "finite and >= 0"
        )

    return result


def _vector(
    value: object,
    name: str,
    length: int,
    *,
    positive: bool = False,
) -> tuple[float, ...]:

    if not isinstance(
        value,
        (list, tuple),
    ):
        raise ValueError(
            f"{name} must be a list"
        )

    result = tuple(
        float(item)
        for item in value
    )

    if len(result) != length:
        raise ValueError(
            f"{name} must contain "
            f"{length} values"
        )

    if not all(
        math.isfinite(item)
        for item in result
    ):
        raise ValueError(
            f"{name} contains "
            "NaN or Inf"
        )

    if (
        positive
        and not all(
            item > 0
            for item in result
        )
    ):
        raise ValueError(
            f"{name} must contain "
            "positive values"
        )

    return result


def _resolve_optional_path(
    project_root: Path,
    value: object | None,
) -> Path | None:

    if value is None:
        return None

    path = Path(
        str(value)
    ).expanduser()

    if not path.is_absolute():
        path = (
            project_root
            / path
        )

    return path


def _resolve_path(
    project_root: Path,
    value: object,
) -> Path:

    result = (
        _resolve_optional_path(
            project_root,
            value,
        )
    )

    if result is None:
        raise ValueError(
            "path must not be null"
        )

    return result


def load_run_config(
    path: str | Path,
) -> RunConfig:

    path = Path(path)

    project_root = (
        path.resolve()
        .parents[1]
    )

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
            "run config root "
            "must be a mapping"
        )

    act_raw = raw.get("act")

    insertion_raw = raw.get(
        "insertion"
    )

    if not isinstance(
        act_raw,
        dict,
    ):
        raise ValueError(
            "'act' must be a mapping"
        )

    if not isinstance(
        insertion_raw,
        dict,
    ):
        raise ValueError(
            "'insertion' must be "
            "a mapping"
        )

    control_fps = (
        _positive_float(
            raw[
                "control_fps"
            ],
            "control_fps",
        )
    )

    home_q = _vector(
        raw["home_q"],
        "home_q",
        6,
    )

    home_duration_s = (
        _positive_float(
            raw[
                "home_duration_s"
            ],
            "home_duration_s",
        )
    )

    max_attempts = int(
        raw["max_attempts"]
    )

    if max_attempts <= 0:
        raise ValueError(
            "max_attempts must be > 0"
        )

    visual_alignment_timeout_s = (
        _positive_float(
            raw[
                "visual_alignment_timeout_s"
            ],
            (
                "visual_alignment_"
                "timeout_s"
            ),
        )
    )

    max_delta_per_step = _vector(
        raw[
            "max_delta_per_step"
        ],
        "max_delta_per_step",
        6,
        positive=True,
    )

    checkpoint = str(
        act_raw[
            "checkpoint"
        ]
    ).strip()

    dataset_repo_id = str(
        act_raw[
            "dataset_repo_id"
        ]
    ).strip()

    task = str(
        act_raw["task"]
    ).strip()

    device = str(
        act_raw.get(
            "device",
            "cuda",
        )
    ).strip()

    if not checkpoint:
        raise ValueError(
            "act.checkpoint "
            "must not be empty"
        )

    if not dataset_repo_id:
        raise ValueError(
            "act.dataset_repo_id "
            "must not be empty"
        )

    if not task:
        raise ValueError(
            "act.task must not be empty"
        )

    if not device:
        raise ValueError(
            "act.device must not be empty"
        )

    act = ACTRunConfig(
        checkpoint=checkpoint,
        dataset_repo_id=(
            dataset_repo_id
        ),
        dataset_root=(
            _resolve_optional_path(
                project_root,
                act_raw.get(
                    "dataset_root"
                ),
            )
        ),
        task=task,
        device=device,
        timeout_s=_positive_float(
            act_raw[
                "timeout_s"
            ],
            "act.timeout_s",
        ),
    )

    correction_error_keys = tuple(
        str(value).strip()
        for value in insertion_raw[
            "correction_error_keys"
        ]
    )

    if (
        not correction_error_keys
        or any(
            not key
            for key
            in correction_error_keys
        )
    ):
        raise ValueError(
            "insertion."
            "correction_error_keys "
            "must not be empty"
        )

    raw_correction_delta = (
        insertion_raw[
            "correction_max_joint_delta"
        ]
    )

    if not isinstance(
        raw_correction_delta,
        (list, tuple),
    ):
        raise ValueError(
            "insertion."
            "correction_max_joint_delta "
            "must be a list"
        )

    correction_max_joint_delta = tuple(
        float(value)
        for value
        in raw_correction_delta
    )

    if not (
        correction_max_joint_delta
    ):
        raise ValueError(
            "correction_max_joint_delta "
            "must not be empty"
        )

    if not all(
        math.isfinite(value)
        and value > 0
        for value in (
            correction_max_joint_delta
        )
    ):
        raise ValueError(
            "correction_max_joint_delta "
            "must contain finite "
            "positive values"
        )

    guard_raw = insertion_raw.get(
        "guard_error_limits"
    )

    if (
        not isinstance(
            guard_raw,
            dict,
        )
        or not guard_raw
    ):
        raise ValueError(
            "guard_error_limits "
            "must be a non-empty mapping"
        )

    guard_error_limits = {
        str(key): _positive_float(
            value,
            (
                "insertion."
                "guard_error_limits."
                f"{key}"
            ),
        )
        for key, value
        in guard_raw.items()
    }

    progress_step = _positive_float(
        insertion_raw[
            "progress_step"
        ],
        "insertion.progress_step",
    )

    if progress_step > 1:
        raise ValueError(
            "insertion.progress_step "
            "must be <= 1"
        )

    insertion = (
        InsertionRunConfig(
            joint_delta_path=(
                _resolve_path(
                    project_root,
                    insertion_raw[
                        "joint_delta_path"
                    ],
                )
            ),
            success_reference_path=(
                _resolve_path(
                    project_root,
                    insertion_raw[
                        "success_reference_path"
                    ],
                )
            ),
            progress_step=(
                progress_step
            ),
            timeout_s=_positive_float(
                insertion_raw[
                    "timeout_s"
                ],
                "insertion.timeout_s",
            ),
            correction_error_keys=(
                correction_error_keys
            ),
            correction_gain=(
                _positive_float(
                    insertion_raw[
                        "correction_gain"
                    ],
                    (
                        "insertion."
                        "correction_gain"
                    ),
                )
            ),
            correction_damping=(
                _nonnegative_float(
                    insertion_raw[
                        "correction_damping"
                    ],
                    (
                        "insertion."
                        "correction_damping"
                    ),
                )
            ),
            correction_max_joint_delta=(
                correction_max_joint_delta
            ),
            guard_error_limits=(
                guard_error_limits
            ),
            retract_scale=(
                _positive_float(
                    insertion_raw[
                        "retract_scale"
                    ],
                    (
                        "insertion."
                        "retract_scale"
                    ),
                )
            ),
            retract_duration_s=(
                _positive_float(
                    insertion_raw[
                        "retract_duration_s"
                    ],
                    (
                        "insertion."
                        "retract_duration_s"
                    ),
                )
            ),
        )
    )

    return RunConfig(
        control_fps=control_fps,
        home_q=home_q,
        home_duration_s=(
            home_duration_s
        ),
        max_attempts=max_attempts,
        visual_alignment_timeout_s=(
            visual_alignment_timeout_s
        ),
        max_delta_per_step=(
            max_delta_per_step
        ),
        act=act,
        insertion=insertion,
    )
