from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(slots=True)
class RobotState:
    """
    机器人当前状态。
    """

    q: np.ndarray
    timestamp_s: float


@dataclass(slots=True)
class FrameBundle:
    """
    同一控制周期内的多相机图像。
    """

    frames: Mapping[str, np.ndarray]
    timestamp_s: float


@dataclass(slots=True)
class Observation:
    """
    一个控制周期内的完整观测。
    """

    robot: RobotState
    cameras: FrameBundle


@dataclass(slots=True)
class Action:
    """
    机器人目标动作。
    """

    q: np.ndarray
    timestamp_s: float | None = None


@dataclass(slots=True)
class DetectionResult:
    """
    USB 插入成功检测结果。
    """

    success: bool
    score: float | None = None
    consecutive: int = 0
    reason: str = ""


@dataclass(slots=True)
class AlignmentEstimate:
    """
    USB 与插口之间的视觉对准估计。

    Vision 只负责测量，不直接控制机器人。

    visible:
        是否可靠看到了 USB 和插口。

    in_capture_region:
        USB 是否已经被 ACT 搬运到 Visual Servo
        能够接管的区域。

    aligned:
        横向位置与姿态误差是否已经满足
        PREINSERT 要求。

        注意：
        插入轴方向距离不一定包含在 aligned 判断中，
        因为进入插入阶段后 USB 会沿插入轴继续移动。

    errors:
        视觉误差。

        真实系统以后可能包括：
            top_dx_px
            top_dy_px
            side_dx_px
            side_dy_px
            yaw_error_deg
            pitch_error_deg
            roll_error_deg

    confidence:
        感知可信度，范围计划为 [0, 1]。

    plug_face:
        USB 正反面识别结果。

        后续可以是：
            "marked"
            "unmarked"

        你的 USB 有明显正反标识，
        后面的真实 Vision 会利用这一点。

    reason:
        调试信息。
    """

    visible: bool
    in_capture_region: bool
    aligned: bool

    errors: Mapping[str, float] = field(
        default_factory=dict
    )

    confidence: float = 0.0
    plug_face: str | None = None
    reason: str = ""