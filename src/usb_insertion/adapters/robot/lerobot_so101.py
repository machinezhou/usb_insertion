from __future__ import annotations

import time
from typing import Any

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    FrameBundle,
    Observation,
    RobotState,
)


SO101_JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

SO101_CAMERA_KEYS = (
    "top",
    "wrist",
    "side",
)


class LeRobotSO101Adapter:
    """
    LeRobot SO101Follower -> usb_insertion adapter。

    当前 follower 配置：
        use_degrees=True

    因此 body joints 使用 LeRobot calibrated degree 表示；
    gripper 保持 LeRobot 自己的 calibrated position 表示。
    """

    def __init__(
        self,
        robot: Any,
        calibrate_on_connect: bool = False,
    ):
        self._robot = robot
        self._calibrate_on_connect = (
            calibrate_on_connect
        )

    @property
    def raw_robot(self) -> Any:
        """
        只供 diagnostics / calibration 工具使用。
        正常业务层不要绕过 Adapter。
        """
        return self._robot

    def connect(self) -> None:
        self._robot.connect(
            calibrate=self._calibrate_on_connect
        )

    def close(self) -> None:
        self._robot.disconnect()

    def get_state(self) -> RobotState:
        """
        只读取电机位置，不读取 cameras。

        这对于：
        - scripted motion
        - retract
        - trajectory start state

        比调用完整 get_observation() 更合适。
        """

        raw_state = (
            self._robot
            .bus
            .sync_read(
                "Present_Position",
                num_retry=(
                    self._robot
                    .config
                    .num_read_retries
                ),
            )
        )

        raw = {
            f"{motor}.pos": value
            for motor, value
            in raw_state.items()
        }

        q = self._extract_joint_array(
            raw
        )

        return RobotState(
            q=q,
            timestamp_s=time.monotonic(),
        )

    def get_observation(
        self,
    ) -> Observation:
        raw = (
            self._robot
            .get_observation()
        )

        timestamp_s = (
            time.monotonic()
        )

        robot_state = RobotState(
            q=self._extract_joint_array(
                raw
            ),
            timestamp_s=timestamp_s,
        )

        frames = FrameBundle(
            frames=self._extract_frames(
                raw
            ),
            timestamp_s=timestamp_s,
        )

        return Observation(
            robot=robot_state,
            cameras=frames,
        )

    def send_action(
        self,
        action: Action,
    ) -> None:
        q = np.asarray(
            action.q,
            dtype=np.float64,
        )

        if q.ndim != 1:
            raise ValueError(
                "action.q must be 1-D"
            )

        if q.shape != (6,):
            raise ValueError(
                "SO-101 action must have "
                f"shape (6,), got {q.shape}"
            )

        if not np.all(
            np.isfinite(q)
        ):
            raise ValueError(
                "action.q contains NaN or Inf"
            )

        raw_action = {
            key: float(value)
            for key, value
            in zip(
                SO101_JOINT_KEYS,
                q,
                strict=True,
            )
        }

        self._robot.send_action(
            raw_action
        )

    @staticmethod
    def _extract_joint_array(
        raw: dict[str, Any],
    ) -> np.ndarray:
        missing = [
            key
            for key in SO101_JOINT_KEYS
            if key not in raw
        ]

        if missing:
            raise KeyError(
                "Missing SO-101 joint keys: "
                f"{missing}"
            )

        q = np.asarray(
            [
                raw[key]
                for key
                in SO101_JOINT_KEYS
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(q)
        ):
            raise ValueError(
                "SO-101 state contains "
                "NaN or Inf"
            )

        return q

    @staticmethod
    def _extract_frames(
        raw: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        frames = {}

        for key in SO101_CAMERA_KEYS:
            if key not in raw:
                raise KeyError(
                    "Missing camera: "
                    f"{key}"
                )

            frame = raw[key]

            if not isinstance(
                frame,
                np.ndarray,
            ):
                raise TypeError(
                    f"{key} is not ndarray"
                )

            if frame.ndim != 3:
                raise ValueError(
                    f"{key} must be HxWxC"
                )

            frames[key] = frame

        return frames
