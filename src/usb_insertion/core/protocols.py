from __future__ import annotations

from typing import Protocol

from usb_insertion.core.datatypes import (
    Action,
    AlignmentEstimate,
    DetectionResult,
    FrameBundle,
    Observation,
    RobotState,
)


class RobotBackend(Protocol):
    def connect(self) -> None:
        ...

    def close(self) -> None:
        ...

    def get_state(self) -> RobotState:
        ...

    def send_action(
        self,
        action: Action,
    ) -> None:
        ...


class CameraBackend(Protocol):
    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def read(self) -> FrameBundle:
        ...


class ObservationSource(Protocol):
    def get_observation(
        self,
    ) -> Observation:
        ...


class PolicyBackend(Protocol):
    """
    ACT 等学习策略统一接口。
    """

    def reset(self) -> None:
        ...

    def predict(
        self,
        observation: Observation,
    ) -> Action:
        ...


class AlignmentEstimator(Protocol):
    """
    Vision 感知层。

    只负责：
    - USB 是否可见
    - 是否进入 Visual Servo capture region
    - USB 相对理想 pose 的图像误差
    - 是否达到 pre-insert 对准容差

    不负责控制机器人。
    """

    def reset(self) -> None:
        ...

    def estimate(
        self,
        observation: Observation,
    ) -> AlignmentEstimate:
        ...


class VisualServoController(Protocol):
    """
    视觉伺服控制器。
    """

    def reset(self) -> None:
        ...

    def compute_action(
        self,
        observation: Observation,
        alignment: AlignmentEstimate,
    ) -> Action:
        ...


class VisualInsertionController(Protocol):
    """
    最终插入控制器。

    is_safe():
        当前视觉误差是否还允许继续向前插。

    compute_action():
        生成下一周期：
        插入方向前进 + 视觉纠偏。
    """

    def reset(self) -> None:
        ...

    def is_safe(
        self,
        alignment: AlignmentEstimate,
    ) -> bool:
        ...

    def compute_action(
        self,
        observation: Observation,
        alignment: AlignmentEstimate,
    ) -> Action:
        ...


class SuccessDetector(Protocol):
    """
    USB 插入成功检测器。

    alignment 可选：
        Fake detector 可以忽略它；
        真实 detector 直接复用当前周期已经算出的
        AlignmentEstimate，避免重复运行视觉特征检测。
    """

    def reset(self) -> None:
        ...

    def update(
        self,
        frames: FrameBundle,
        alignment: AlignmentEstimate | None = None,
    ) -> DetectionResult:
        ...
