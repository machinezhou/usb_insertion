from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from usb_insertion.adapters.policy.lerobot_act import (
    LeRobotACTAdapter,
)
from usb_insertion.adapters.robot.lerobot_so101 import (
    SO101_JOINT_KEYS,
)
from usb_insertion.core.datatypes import (
    FrameBundle,
    Observation,
    RobotState,
)


class DummyPolicy:
    def __init__(self):
        self.reset_count = 0
        self.select_calls = []

    def reset(self) -> None:
        self.reset_count += 1

    def select_action(
        self,
        frame,
    ):
        self.select_calls.append(
            frame
        )

        return torch.tensor(
            [
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ]
            ],
            dtype=torch.float32,
        )


class DummyProcessor:
    def __init__(
        self,
        prefix: str,
    ):
        self.prefix = prefix
        self.reset_count = 0
        self.calls = []

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(
        self,
        value,
    ):
        self.calls.append(
            value
        )

        return {
            self.prefix: value
        }


def make_dataset_features():
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": [
                6
            ],
            "names": list(
                SO101_JOINT_KEYS
            ),
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": [
                3,
                480,
                640,
            ],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [
                3,
                480,
                640,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [
                6
            ],
            "names": list(
                SO101_JOINT_KEYS
            ),
        },
    }


def make_observation():
    timestamp_s = 123.456

    return Observation(
        robot=RobotState(
            q=np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ],
                dtype=np.float64,
            ),
            timestamp_s=timestamp_s,
        ),
        cameras=FrameBundle(
            frames={
                "top": np.zeros(
                    (
                        480,
                        640,
                        3,
                    ),
                    dtype=np.uint8,
                ),
                "wrist": np.zeros(
                    (
                        480,
                        640,
                        3,
                    ),
                    dtype=np.uint8,
                ),
                "side": np.zeros(
                    (
                        480,
                        640,
                        3,
                    ),
                    dtype=np.uint8,
                ),
            },
            timestamp_s=timestamp_s,
        ),
    )


class TestLeRobotACTAdapter(
    unittest.TestCase
):
    def make_adapter(
        self,
        dataset_features=None,
    ):
        if dataset_features is None:
            dataset_features = (
                make_dataset_features()
            )

        policy = DummyPolicy()

        preprocessor = (
            DummyProcessor(
                "preprocessed"
            )
        )

        postprocessor = (
            DummyProcessor(
                "postprocessed"
            )
        )

        adapter = LeRobotACTAdapter(
            policy=policy,
            preprocessor=(
                preprocessor
            ),
            postprocessor=(
                postprocessor
            ),
            dataset_features=(
                dataset_features
            ),
            device=torch.device(
                "cpu"
            ),
            task=(
                "Pick up the USB plug "
                "and move it near "
                "the socket"
            ),
            robot_type=(
                "so101_follower"
            ),
        )

        return (
            adapter,
            policy,
            preprocessor,
            postprocessor,
        )

    def test_camera_names_are_read_from_dataset(
        self,
    ):
        (
            adapter,
            _,
            _,
            _,
        ) = self.make_adapter()

        self.assertEqual(
            adapter.camera_names,
            (
                "top",
                "wrist",
            ),
        )

    def test_dataset_action_order_must_match_so101(
        self,
    ):
        features = (
            make_dataset_features()
        )

        features[
            "action"
        ][
            "names"
        ] = list(
            reversed(
                SO101_JOINT_KEYS
            )
        )

        with self.assertRaises(
            ValueError
        ):
            self.make_adapter(
                features
            )

    def test_dataset_state_order_must_match_so101(
        self,
    ):
        features = (
            make_dataset_features()
        )

        features[
            "observation.state"
        ][
            "names"
        ] = list(
            reversed(
                SO101_JOINT_KEYS
            )
        )

        with self.assertRaises(
            ValueError
        ):
            self.make_adapter(
                features
            )

    def test_dataset_requires_camera_feature(
        self,
    ):
        features = {
            key: value
            for key, value
            in make_dataset_features()
            .items()
            if not key.startswith(
                "observation.images."
            )
        }

        with self.assertRaises(
            ValueError
        ):
            self.make_adapter(
                features
            )

    def test_missing_expected_camera_fails(
        self,
    ):
        (
            adapter,
            _,
            _,
            _,
        ) = self.make_adapter()

        observation = (
            make_observation()
        )

        observation.cameras.frames = {
            "top": (
                observation
                .cameras
                .frames[
                    "top"
                ]
            )
        }

        with self.assertRaises(
            KeyError
        ):
            adapter._to_raw_observation(
                observation
            )

    def test_invalid_joint_shape_fails(
        self,
    ):
        (
            adapter,
            _,
            _,
            _,
        ) = self.make_adapter()

        observation = (
            make_observation()
        )

        observation.robot.q = (
            np.zeros(
                5,
                dtype=np.float64,
            )
        )

        with self.assertRaises(
            ValueError
        ):
            adapter._to_raw_observation(
                observation
            )

    def test_nonfinite_joint_state_fails(
        self,
    ):
        (
            adapter,
            _,
            _,
            _,
        ) = self.make_adapter()

        observation = (
            make_observation()
        )

        observation.robot.q[
            2
        ] = np.nan

        with self.assertRaises(
            ValueError
        ):
            adapter._to_raw_observation(
                observation
            )

    def test_predict_runs_complete_pipeline(
        self,
    ):
        (
            adapter,
            policy,
            preprocessor,
            postprocessor,
        ) = self.make_adapter()

        observation = (
            make_observation()
        )

        robot_action = {
            key: float(
                index + 10
            )
            for index, key
            in enumerate(
                SO101_JOINT_KEYS
            )
        }

        with (
            patch(
                (
                    "usb_insertion.adapters."
                    "policy.lerobot_act."
                    "build_inference_frame"
                ),
                return_value={
                    "frame": "raw"
                },
            ) as build_frame,
            patch(
                (
                    "usb_insertion.adapters."
                    "policy.lerobot_act."
                    "make_robot_action"
                ),
                return_value=(
                    robot_action
                ),
            ) as make_action,
        ):
            action = adapter.predict(
                observation
            )

        build_frame.assert_called_once()

        self.assertEqual(
            len(
                preprocessor.calls
            ),
            1,
        )

        self.assertEqual(
            len(
                policy.select_calls
            ),
            1,
        )

        self.assertEqual(
            len(
                postprocessor.calls
            ),
            1,
        )

        make_action.assert_called_once()

        np.testing.assert_allclose(
            action.q,
            np.array(
                [
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ],
                dtype=np.float64,
            ),
        )

        self.assertEqual(
            action.timestamp_s,
            observation
            .robot
            .timestamp_s,
        )

    def test_missing_action_key_fails(
        self,
    ):
        (
            adapter,
            _,
            _,
            _,
        ) = self.make_adapter()

        observation = (
            make_observation()
        )

        incomplete_action = {
            key: float(index)
            for index, key
            in enumerate(
                SO101_JOINT_KEYS[
                    :-1
                ]
            )
        }

        with (
            patch(
                (
                    "usb_insertion.adapters."
                    "policy.lerobot_act."
                    "build_inference_frame"
                ),
                return_value={
                    "frame": "raw"
                },
            ),
            patch(
                (
                    "usb_insertion.adapters."
                    "policy.lerobot_act."
                    "make_robot_action"
                ),
                return_value=(
                    incomplete_action
                ),
            ),
        ):
            with self.assertRaises(
                KeyError
            ):
                adapter.predict(
                    observation
                )

    def test_reset_resets_entire_pipeline(
        self,
    ):
        (
            adapter,
            policy,
            preprocessor,
            postprocessor,
        ) = self.make_adapter()

        adapter.reset()

        self.assertEqual(
            policy.reset_count,
            1,
        )

        self.assertEqual(
            preprocessor.reset_count,
            1,
        )

        self.assertEqual(
            postprocessor.reset_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
