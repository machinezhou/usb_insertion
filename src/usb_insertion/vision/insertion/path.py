from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(
    frozen=True,
    slots=True,
)
class InsertionPose:
    stage: str
    q: np.ndarray
    timestamp_s: float | None = None


def _validate_q(
    q: np.ndarray,
) -> np.ndarray:

    array = np.asarray(
        q,
        dtype=np.float64,
    )

    if array.shape != (6,):
        raise ValueError(
            "SO-101 insertion pose "
            "must have shape (6,), "
            f"got {array.shape}"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "SO-101 insertion pose "
            "contains NaN or Inf"
        )

    return array.copy()


def save_insertion_pose(
    path: str | Path,
    pose: InsertionPose,
) -> Path:

    path = Path(path)

    q = _validate_q(
        pose.q
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "stage": str(
            pose.stage
        ),
        "q": q.tolist(),
        "timestamp_s": (
            None
            if pose.timestamp_s
            is None
            else float(
                pose.timestamp_s
            )
        ),
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path


def load_insertion_pose(
    path: str | Path,
) -> InsertionPose:

    path = Path(path)

    raw = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            "insertion pose root "
            "must be an object"
        )

    stage = str(
        raw.get(
            "stage",
            "",
        )
    )

    if not stage:
        raise ValueError(
            "insertion pose stage "
            "must not be empty"
        )

    q = _validate_q(
        np.asarray(
            raw["q"],
            dtype=np.float64,
        )
    )

    timestamp_raw = raw.get(
        "timestamp_s"
    )

    timestamp_s = (
        None
        if timestamp_raw is None
        else float(
            timestamp_raw
        )
    )

    return InsertionPose(
        stage=stage,
        q=q,
        timestamp_s=timestamp_s,
    )


def build_insertion_joint_delta(
    preinsert_q: np.ndarray,
    inserted_q: np.ndarray,
    *,
    max_gripper_delta: float = 1.0,
) -> np.ndarray:

    if max_gripper_delta < 0:
        raise ValueError(
            "max_gripper_delta "
            "must be >= 0"
        )

    preinsert = _validate_q(
        preinsert_q
    )

    inserted = _validate_q(
        inserted_q
    )

    raw_delta = (
        inserted
        - preinsert
    )

    if abs(
        float(
            raw_delta[5]
        )
    ) > max_gripper_delta:
        raise ValueError(
            "gripper position changed "
            "too much between "
            "preinsert and inserted: "
            f"{raw_delta[5]}"
        )

    delta = raw_delta.copy()

    # 插入过程中绝不改变 ACT
    # 已经建立好的抓取夹爪位置。
    delta[5] = 0.0

    if (
        np.linalg.norm(
            delta[:5]
        )
        <= 0
    ):
        raise ValueError(
            "insertion joint delta "
            "is zero"
        )

    return delta


def save_insertion_joint_delta(
    path: str | Path,
    delta: np.ndarray,
    *,
    preinsert_pose_path: (
        str | Path | None
    ) = None,
    inserted_pose_path: (
        str | Path | None
    ) = None,
) -> Path:

    path = Path(path)

    if path.suffix != ".npy":
        raise ValueError(
            "insertion joint delta "
            "must use .npy"
        )

    array = _validate_q(
        delta
    )

    if abs(
        float(
            array[5]
        )
    ) > 1e-9:
        raise ValueError(
            "insertion joint delta "
            "gripper component "
            "must be zero"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        array,
        allow_pickle=False,
    )

    metadata = {
        "joint_delta": (
            array.tolist()
        ),
        "preinsert_pose_path": (
            None
            if preinsert_pose_path
            is None
            else str(
                preinsert_pose_path
            )
        ),
        "inserted_pose_path": (
            None
            if inserted_pose_path
            is None
            else str(
                inserted_pose_path
            )
        ),
    }

    path.with_suffix(
        ".json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path
