from __future__ import annotations

from dataclasses import dataclass

from usb_insertion.workflow.states import (
    TaskEvent,
    TaskState,
)


@dataclass(slots=True)
class TaskStateMachine:
    """
    新版 USB 任务状态机。

    attempts 表示：
        真正的 USB insertion 尝试次数。

    ACT pick/approach 不计入 attempts。
    """

    max_attempts: int = 2

    state: TaskState = TaskState.INIT
    attempts: int = 0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError(
                "max_attempts must be > 0"
            )

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            TaskState.DONE,
            TaskState.FAILED,
        }

    @property
    def attempts_remaining(self) -> int:
        return max(
            0,
            self.max_attempts
            - self.attempts,
        )

    def dispatch(
        self,
        event: TaskEvent,
    ) -> TaskState:

        if event is TaskEvent.ABORT:
            if not self.is_terminal:
                self.state = (
                    TaskState.FAILED
                )

            return self.state

        if self.is_terminal:
            raise RuntimeError(
                "Task already finished: "
                f"{self.state.name}"
            )

        current = self.state

        if (
            current is TaskState.INIT
            and event is TaskEvent.START
        ):
            self.state = (
                TaskState.HOMING
            )

        elif (
            current is TaskState.HOMING
            and event
            is TaskEvent.HOME_REACHED
        ):
            self.state = (
                TaskState.ACT_PICK_APPROACH
            )

        elif (
            current
            is TaskState.ACT_PICK_APPROACH
            and event
            is TaskEvent.ACT_CAPTURE_REACHED
        ):
            self.state = (
                TaskState.VISUAL_ALIGNING
            )

        elif (
            current
            is TaskState.ACT_PICK_APPROACH
            and event
            is TaskEvent.ACT_APPROACH_FAILED
        ):
            self.state = (
                TaskState.FAILED
            )

        elif (
            current
            is TaskState.VISUAL_ALIGNING
            and event
            is TaskEvent.VISUAL_ALIGNED
        ):
            self.state = (
                TaskState.PREINSERT
            )

        elif (
            current
            is TaskState.VISUAL_ALIGNING
            and event
            is TaskEvent.VISUAL_ALIGNMENT_FAILED
        ):
            self.state = (
                TaskState.FAILED
            )

        elif (
            current
            is TaskState.PREINSERT
            and event
            is TaskEvent.START_INSERTION
        ):
            if (
                self.attempts
                >= self.max_attempts
            ):
                raise RuntimeError(
                    "No insertion attempts remaining"
                )

            self.attempts += 1

            self.state = (
                TaskState.VISUAL_INSERTING
            )

        elif (
            current
            is TaskState.VISUAL_INSERTING
            and event
            is TaskEvent.INSERTION_FINISHED
        ):
            self.state = (
                TaskState.VERIFY
            )

        elif (
            current is TaskState.VERIFY
            and event
            is TaskEvent.VERIFY_SUCCESS
        ):
            self.state = (
                TaskState.DONE
            )

        elif (
            current is TaskState.VERIFY
            and event
            is TaskEvent.VERIFY_FAILURE
        ):
            if (
                self.attempts
                < self.max_attempts
            ):
                self.state = (
                    TaskState.RETRACTING
                )
            else:
                self.state = (
                    TaskState.FAILED
                )

        elif (
            current
            is TaskState.RETRACTING
            and event
            is TaskEvent.RETRACT_FINISHED
        ):
            # Retry 不重新跑 ACT。
            #
            # USB 已经在夹爪里，
            # 先退开再重新视觉对准。
            self.state = (
                TaskState.VISUAL_ALIGNING
            )

        else:
            raise RuntimeError(
                "Invalid transition: "
                f"{current.name} + {event.name}"
            )

        return self.state