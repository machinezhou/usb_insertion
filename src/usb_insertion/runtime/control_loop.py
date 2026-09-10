from __future__ import annotations

from dataclasses import dataclass

from usb_insertion.control.action_guard import ActionGuard
from usb_insertion.core.datatypes import (
    Action,
    Observation,
)
from usb_insertion.core.protocols import (
    ObservationSource,
    PolicyBackend,
    RobotBackend,
)


@dataclass(slots=True)
class ControlStepResult:
    """
    一次 policy control step 的完整结果。

    保留：
    - 输入 observation
    - policy 原始输出
    - ActionGuard 后的安全输出
    """

    observation: Observation
    raw_action: Action
    safe_action: Action


def run_policy_step_from_observation(
    observation: Observation,
    policy: PolicyBackend,
    robot: RobotBackend,
    action_guard: ActionGuard,
) -> ControlStepResult:
    """
    使用已经取得的 Observation 执行一次 policy step。

    数据流：

        Observation
            ↓
          Policy
            ↓
        raw Action
            ↓
       ActionGuard
            ↓
       safe Action
            ↓
          Robot
    """

    raw_action = policy.predict(
        observation
    )

    safe_action = action_guard.apply(
        raw_action=raw_action,
        current_q=observation.robot.q,
    )

    robot.send_action(
        safe_action
    )

    return ControlStepResult(
        observation=observation,
        raw_action=raw_action,
        safe_action=safe_action,
    )


def run_policy_step(
    observation_source: ObservationSource,
    policy: PolicyBackend,
    robot: RobotBackend,
    action_guard: ActionGuard,
) -> ControlStepResult:
    """
    读取最新 Observation，然后执行一次 policy step。

    这是方便普通调用的包装函数。

    Orchestrator 后面会先取得 Observation 做 success check，
    如果还没成功，再调用 run_policy_step_from_observation()。
    """

    observation = (
        observation_source.get_observation()
    )

    return run_policy_step_from_observation(
        observation=observation,
        policy=policy,
        robot=robot,
        action_guard=action_guard,
    )