from __future__ import annotations

import time

import numpy as np

from usb_insertion.core.datatypes import (
    Action,
    Observation,
)


class FakePolicy:
    """
    用于软件测试的确定性假策略。

    它不模拟 ACT 网络。

    每次 predict() 都让当前关节位置朝固定 target_q
    前进一定比例：

        q_next = q_current
                 + gain * (target_q - q_current)

    用途：
    - 测试 Observation -> Action 数据流
    - 测试 ActionGuard
    - 测试 orchestrator
    - 测试 retry / policy reset
    """

    def __init__(
        self,
        target_q: np.ndarray,
        gain: float = 0.25,
    ):
        target_q = np.asarray(
            target_q,
            dtype=np.float64,
        )

        if target_q.ndim != 1:
            raise ValueError(
                "target_q must be a 1-D array"
            )

        if not np.all(np.isfinite(target_q)):
            raise ValueError(
                "target_q contains NaN or Inf"
            )

        if not 0.0 < gain <= 1.0:
            raise ValueError(
                "gain must be in the range (0, 1]"
            )

        self._target_q = target_q.copy()
        self._gain = gain

        self._predict_calls = 0
        self._episode_step = 0

    @property
    def predict_calls(self) -> int:
        """
        该对象生命周期内累计调用 predict() 的次数。
        """
        return self._predict_calls

    @property
    def episode_step(self) -> int:
        """
        当前 episode / attempt 内已经执行的 predict 次数。

        reset() 后归零。
        """
        return self._episode_step

    def reset(self) -> None:
        """
        模拟真实 policy 的运行时状态清空。

        注意：
        predict_calls 是生命周期统计，不清零；
        episode_step 属于当前 attempt 状态，需要清零。
        """
        self._episode_step = 0

    def predict(
        self,
        observation: Observation,
    ) -> Action:
        current_q = np.asarray(
            observation.robot.q,
            dtype=np.float64,
        )

        if current_q.ndim != 1:
            raise ValueError(
                "observation.robot.q must be a 1-D array"
            )

        if current_q.shape != self._target_q.shape:
            raise ValueError(
                "robot state and policy target must "
                "have the same shape: "
                f"{current_q.shape} != "
                f"{self._target_q.shape}"
            )

        if not np.all(np.isfinite(current_q)):
            raise ValueError(
                "observation.robot.q contains NaN or Inf"
            )

        delta = self._target_q - current_q

        next_q = (
            current_q
            + self._gain * delta
        )

        self._predict_calls += 1
        self._episode_step += 1

        return Action(
            q=next_q,
            timestamp_s=time.monotonic(),
        )