from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.dataset_metadata import (
    LeRobotDatasetMetadata,
)
from lerobot.policies.act.modeling_act import (
    ACTPolicy,
)
from lerobot.policies.factory import (
    make_pre_post_processors,
)
from lerobot.policies.utils import (
    build_inference_frame,
    make_robot_action,
)

from usb_insertion.adapters.robot.lerobot_so101 import (
    SO101_JOINT_KEYS,
)
from usb_insertion.core.datatypes import (
    Action,
    Observation,
)


class LeRobotACTAdapter:
    """
    usb_insertion Observation
            ↓
    LeRobot build_inference_frame
            ↓
    checkpoint preprocessor
            ↓
    ACTPolicy.select_action
            ↓
    checkpoint postprocessor
            ↓
    make_robot_action
            ↓
    usb_insertion Action
    """

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        dataset_features: dict,
        device: torch.device,
        task: str,
        robot_type: str = (
            "so101_follower"
        ),
    ):
        self._policy = policy
        self._preprocessor = (
            preprocessor
        )
        self._postprocessor = (
            postprocessor
        )
        self._dataset_features = (
            dataset_features
        )
        self._device = device
        self._task = task
        self._robot_type = (
            robot_type
        )

        self._validate_dataset_features()

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint: str | Path,
        dataset_repo_id: str,
        dataset_root: (
            str | Path | None
        ),
        device: str = "cuda",
        task: str,
        robot_type: str = (
            "so101_follower"
        ),
    ) -> "LeRobotACTAdapter":

        device_obj = torch.device(
            device
        )

        if (
            device_obj.type == "cuda"
            and not (
                torch.cuda.is_available()
            )
        ):
            raise RuntimeError(
                "ACT requested CUDA, "
                "but CUDA is unavailable"
            )

        checkpoint_str = str(
            checkpoint
        )

        metadata = (
            LeRobotDatasetMetadata(
                repo_id=(
                    dataset_repo_id
                ),
                root=(
                    None
                    if dataset_root is None
                    else Path(
                        dataset_root
                    )
                ),
            )
        )

        policy = (
            ACTPolicy
            .from_pretrained(
                checkpoint_str
            )
        )

        policy.to(
            device_obj
        )

        policy.eval()

        preprocessor, postprocessor = (
            make_pre_post_processors(
                policy_cfg=(
                    policy.config
                ),
                pretrained_path=(
                    checkpoint_str
                ),
                preprocessor_overrides={
                    "device_processor": {
                        "device": str(
                            device_obj
                        )
                    }
                },
            )
        )

        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=(
                postprocessor
            ),
            dataset_features=(
                metadata.features
            ),
            device=device_obj,
            task=task,
            robot_type=robot_type,
        )

    @property
    def dataset_features(
        self,
    ) -> dict:
        return self._dataset_features

    @property
    def device(
        self,
    ) -> torch.device:
        return self._device

    @property
    def camera_names(
        self,
    ) -> tuple[str, ...]:

        names = []

        prefix = (
            "observation.images."
        )

        for key, feature in (
            self._dataset_features
            .items()
        ):
            if (
                key.startswith(
                    prefix
                )
                and feature.get(
                    "dtype"
                )
                in (
                    "image",
                    "video",
                )
            ):
                names.append(
                    key.removeprefix(
                        prefix
                    )
                )

        return tuple(
            names
        )

    def reset(self) -> None:
        self._policy.reset()

        if hasattr(
            self._preprocessor,
            "reset",
        ):
            self._preprocessor.reset()

        if hasattr(
            self._postprocessor,
            "reset",
        ):
            self._postprocessor.reset()

    def _validate_dataset_features(
        self,
    ) -> None:

        action_feature = (
            self._dataset_features
            .get("action")
        )

        if not isinstance(
            action_feature,
            dict,
        ):
            raise ValueError(
                "Dataset has no valid "
                "'action' feature"
            )

        action_names = (
            action_feature.get(
                "names"
            )
        )

        if action_names is None:
            raise ValueError(
                "Dataset action has "
                "no names"
            )

        if (
            tuple(action_names)
            != SO101_JOINT_KEYS
        ):
            raise ValueError(
                "ACT action order does "
                "not match SO-101.\n"
                f"dataset="
                f"{tuple(action_names)}\n"
                f"expected="
                f"{SO101_JOINT_KEYS}"
            )

        state_feature = (
            self._dataset_features
            .get(
                "observation.state"
            )
        )

        if not isinstance(
            state_feature,
            dict,
        ):
            raise ValueError(
                "Dataset has no valid "
                "observation.state"
            )

        state_names = (
            state_feature.get(
                "names"
            )
        )

        if state_names is None:
            raise ValueError(
                "observation.state "
                "has no names"
            )

        if (
            tuple(state_names)
            != SO101_JOINT_KEYS
        ):
            raise ValueError(
                "ACT observation.state "
                "order does not match "
                "SO-101"
            )

        if not self.camera_names:
            raise ValueError(
                "ACT dataset contains "
                "no camera feature"
            )

    def _to_raw_observation(
        self,
        observation: Observation,
    ) -> dict[str, Any]:

        q = np.asarray(
            observation.robot.q,
            dtype=np.float64,
        )

        if q.shape != (6,):
            raise ValueError(
                "ACT observation q "
                "must have shape (6,)"
            )

        if not np.all(
            np.isfinite(q)
        ):
            raise ValueError(
                "ACT observation q "
                "contains NaN or Inf"
            )

        raw = {
            key: float(value)
            for key, value
            in zip(
                SO101_JOINT_KEYS,
                q,
                strict=True,
            )
        }

        frames = (
            observation
            .cameras
            .frames
        )

        for camera_name in (
            self.camera_names
        ):
            if camera_name not in (
                frames
            ):
                raise KeyError(
                    "ACT dataset expects "
                    f"camera '{camera_name}'"
                )

            frame = np.asarray(
                frames[
                    camera_name
                ]
            )

            if (
                frame.ndim != 3
                or frame.shape[2]
                not in (
                    1,
                    3,
                )
            ):
                raise ValueError(
                    f"ACT camera "
                    f"'{camera_name}' "
                    "must be HxWxC"
                )

            if (
                np.issubdtype(
                    frame.dtype,
                    np.floating,
                )
                and not np.all(
                    np.isfinite(
                        frame
                    )
                )
            ):
                raise ValueError(
                    f"ACT camera "
                    f"'{camera_name}' "
                    "contains NaN/Inf"
                )

            raw[
                camera_name
            ] = frame

        return raw

    def predict(
        self,
        observation: Observation,
    ) -> Action:

        raw_observation = (
            self._to_raw_observation(
                observation
            )
        )

        frame = (
            build_inference_frame(
                observation=(
                    raw_observation
                ),
                ds_features=(
                    self
                    ._dataset_features
                ),
                device=self._device,
                task=self._task,
                robot_type=(
                    self._robot_type
                ),
            )
        )

        frame = (
            self._preprocessor(
                frame
            )
        )

        with torch.inference_mode():
            policy_action = (
                self._policy
                .select_action(
                    frame
                )
            )

        policy_action = (
            self._postprocessor(
                policy_action
            )
        )

        robot_action = (
            make_robot_action(
                policy_action,
                self._dataset_features,
            )
        )

        missing = [
            key
            for key
            in SO101_JOINT_KEYS
            if key not in (
                robot_action
            )
        ]

        if missing:
            raise KeyError(
                "ACT output missing "
                f"keys: {missing}"
            )

        q = np.asarray(
            [
                robot_action[key]
                for key
                in SO101_JOINT_KEYS
            ],
            dtype=np.float64,
        )

        if (
            q.shape != (6,)
            or not np.all(
                np.isfinite(q)
            )
        ):
            raise ValueError(
                "ACT output must be "
                "finite shape (6,)"
            )

        return Action(
            q=q,
            timestamp_s=(
                observation
                .robot
                .timestamp_s
            ),
        )
