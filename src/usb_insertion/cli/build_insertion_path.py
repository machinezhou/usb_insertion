from __future__ import annotations

from pathlib import Path

from usb_insertion.vision.insertion.path import (
    build_insertion_joint_delta,
    load_insertion_pose,
    save_insertion_joint_delta,
)


def project_root() -> Path:
    return Path(
        __file__
    ).resolve().parents[3]


def main() -> None:
    root = project_root()

    pre_path = (
        root
        / "outputs"
        / "insertion"
        / "preinsert_pose.json"
    )

    inserted_path = (
        root
        / "outputs"
        / "insertion"
        / "inserted_pose.json"
    )

    pre = load_insertion_pose(
        pre_path
    )

    inserted = (
        load_insertion_pose(
            inserted_path
        )
    )

    if (
        pre.stage
        != "preinsert"
    ):
        raise ValueError(
            "wrong preinsert pose"
        )

    if (
        inserted.stage
        != "inserted"
    ):
        raise ValueError(
            "wrong inserted pose"
        )

    delta = (
        build_insertion_joint_delta(
            pre.q,
            inserted.q,
        )
    )

    output = (
        root
        / "outputs"
        / "insertion"
        / "insertion_joint_delta.npy"
    )

    save_insertion_joint_delta(
        output,
        delta,
        preinsert_pose_path=(
            pre_path
        ),
        inserted_pose_path=(
            inserted_path
        ),
    )

    print(
        "delta:",
        delta,
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
