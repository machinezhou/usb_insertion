from __future__ import annotations

import time

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    RobotState,
)


class FakeRobot:
    """
    内存中的假机器人。

    目的：
    - 测试业务流程
    - 测试 ActionGuard
    - 测试 scripted motion
    - 测试状态机与 orchestrator

    它不模拟真实 SO-101 的动力学。
    send_action() 后，关节状态直接变为目标值。
    """

    def __init__(
        self,
        dof: int = 6,
        initial_q: np.ndarray | None = None,
    ):
        if dof <= 0:
            raise ValueError(
                "dof must be greater than 0"
            )

        if initial_q is None:
            q = np.zeros(
                dof,
                dtype=np.float64,
            )
        else:
            q = np.asarray(
                initial_q,
                dtype=np.float64,
            ).copy()

            if q.ndim != 1:
                raise ValueError(
                    "initial_q must be a 1-D array"
                )

            if q.shape != (dof,):
                raise ValueError(
                    "initial_q shape must match dof: "
                    f"{q.shape} != ({dof},)"
                )

            if not np.all(np.isfinite(q)):
                raise ValueError(
                    "initial_q contains NaN or Inf"
                )

        self._dof = dof
        self._q = q
        self._connected = False

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """
        模拟连接机器人。
        """
        self._connected = True

    def close(self) -> None:
        """
        模拟关闭机器人连接。
        """
        self._connected = False

    def get_state(self) -> RobotState:
        """
        返回当前关节状态。
        """
        self._require_connected()

        return RobotState(
            q=self._q.copy(),
            timestamp_s=time.monotonic(),
        )

    def send_action(
        self,
        action: Action,
    ) -> None:
        """
        FakeRobot 没有动力学。

        收到 action 后，直接把内部 q 更新到目标值。
        """
        self._require_connected()

        target_q = np.asarray(
            action.q,
            dtype=np.float64,
        )

        if target_q.ndim != 1:
            raise ValueError(
                "action.q must be a 1-D array"
            )

        if target_q.shape != (self._dof,):
            raise ValueError(
                "action.q shape must match robot dof: "
                f"{target_q.shape} != ({self._dof},)"
            )

        if not np.all(np.isfinite(target_q)):
            raise ValueError(
                "action.q contains NaN or Inf"
            )

        self._q = target_q.copy()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "FakeRobot is not connected"
            )