from __future__ import annotations

import argparse
from pathlib import Path

from usb_insertion.adapters.policy.lerobot_act import (
    LeRobotACTAdapter,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--dataset-repo-id",
        required=True,
    )

    parser.add_argument(
        "--dataset-root",
        default=None,
    )

    parser.add_argument(
        "--task",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    adapter = (
        LeRobotACTAdapter
        .from_pretrained(
            checkpoint=args.checkpoint,
            dataset_repo_id=(
                args.dataset_repo_id
            ),
            dataset_root=(
                None
                if args.dataset_root
                is None
                else Path(
                    args.dataset_root
                )
            ),
            device=args.device,
            task=args.task,
        )
    )

    print(
        "ACT adapter loaded."
    )

    print(
        "device:",
        adapter.device,
    )

    print(
        "dataset features:"
    )

    for key, value in (
        adapter
        .dataset_features
        .items()
    ):
        print(
            f"  {key}: {value}"
        )

    adapter.reset()

    print(
        "ACT reset passed."
    )


if __name__ == "__main__":
    main()
