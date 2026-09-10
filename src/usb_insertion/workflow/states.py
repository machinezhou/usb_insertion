from __future__ import annotations

from enum import Enum, auto


class TaskState(Enum):
    INIT = auto()

    HOMING = auto()

    # ACT：
    # 找 USB、抓取、抬起、搬运到插口附近
    ACT_PICK_APPROACH = auto()

    # Vision + Visual Servo：
    # 精确位置/姿态对准
    VISUAL_ALIGNING = auto()

    # 已满足插入前 tolerance
    PREINSERT = auto()

    # 慢速视觉闭环插入
    VISUAL_INSERTING = auto()

    VERIFY = auto()

    # 失败后后退，再重新视觉对准
    RETRACTING = auto()

    DONE = auto()
    FAILED = auto()


class TaskEvent(Enum):
    START = auto()

    HOME_REACHED = auto()

    ACT_CAPTURE_REACHED = auto()
    ACT_APPROACH_FAILED = auto()

    VISUAL_ALIGNED = auto()
    VISUAL_ALIGNMENT_FAILED = auto()

    START_INSERTION = auto()
    INSERTION_FINISHED = auto()

    VERIFY_SUCCESS = auto()
    VERIFY_FAILURE = auto()

    RETRACT_FINISHED = auto()

    ABORT = auto()