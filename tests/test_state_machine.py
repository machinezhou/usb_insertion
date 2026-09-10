import unittest

from usb_insertion.workflow.state_machine import (
    TaskStateMachine,
)
from usb_insertion.workflow.states import (
    TaskEvent,
    TaskState,
)


class TestTaskStateMachine(
    unittest.TestCase
):
    def reach_preinsert(
        self,
        machine: TaskStateMachine,
    ) -> None:
        machine.dispatch(
            TaskEvent.START
        )

        machine.dispatch(
            TaskEvent.HOME_REACHED
        )

        machine.dispatch(
            TaskEvent.ACT_CAPTURE_REACHED
        )

        machine.dispatch(
            TaskEvent.VISUAL_ALIGNED
        )

        self.assertEqual(
            machine.state,
            TaskState.PREINSERT,
        )

    def test_initial_state(self):
        machine = TaskStateMachine()

        self.assertEqual(
            machine.state,
            TaskState.INIT,
        )

        self.assertEqual(
            machine.attempts,
            0,
        )

    def test_act_stage(self):
        machine = TaskStateMachine()

        machine.dispatch(
            TaskEvent.START
        )

        machine.dispatch(
            TaskEvent.HOME_REACHED
        )

        self.assertEqual(
            machine.state,
            TaskState.ACT_PICK_APPROACH,
        )

        machine.dispatch(
            TaskEvent.ACT_CAPTURE_REACHED
        )

        self.assertEqual(
            machine.state,
            TaskState.VISUAL_ALIGNING,
        )

    def test_visual_alignment(self):
        machine = TaskStateMachine()

        machine.dispatch(
            TaskEvent.START
        )

        machine.dispatch(
            TaskEvent.HOME_REACHED
        )

        machine.dispatch(
            TaskEvent.ACT_CAPTURE_REACHED
        )

        machine.dispatch(
            TaskEvent.VISUAL_ALIGNED
        )

        self.assertEqual(
            machine.state,
            TaskState.PREINSERT,
        )

    def test_first_insertion_success(self):
        machine = TaskStateMachine(
            max_attempts=2
        )

        self.reach_preinsert(
            machine
        )

        machine.dispatch(
            TaskEvent.START_INSERTION
        )

        self.assertEqual(
            machine.attempts,
            1,
        )

        machine.dispatch(
            TaskEvent.INSERTION_FINISHED
        )

        machine.dispatch(
            TaskEvent.VERIFY_SUCCESS
        )

        self.assertEqual(
            machine.state,
            TaskState.DONE,
        )

    def test_retry_returns_to_visual_alignment(
        self,
    ):
        machine = TaskStateMachine(
            max_attempts=2
        )

        self.reach_preinsert(
            machine
        )

        machine.dispatch(
            TaskEvent.START_INSERTION
        )

        machine.dispatch(
            TaskEvent.INSERTION_FINISHED
        )

        machine.dispatch(
            TaskEvent.VERIFY_FAILURE
        )

        self.assertEqual(
            machine.state,
            TaskState.RETRACTING,
        )

        machine.dispatch(
            TaskEvent.RETRACT_FINISHED
        )

        self.assertEqual(
            machine.state,
            TaskState.VISUAL_ALIGNING,
        )

    def test_all_attempts_fail(self):
        machine = TaskStateMachine(
            max_attempts=2
        )

        self.reach_preinsert(
            machine
        )

        # Attempt 1
        machine.dispatch(
            TaskEvent.START_INSERTION
        )
        machine.dispatch(
            TaskEvent.INSERTION_FINISHED
        )
        machine.dispatch(
            TaskEvent.VERIFY_FAILURE
        )
        machine.dispatch(
            TaskEvent.RETRACT_FINISHED
        )

        machine.dispatch(
            TaskEvent.VISUAL_ALIGNED
        )

        # Attempt 2
        machine.dispatch(
            TaskEvent.START_INSERTION
        )
        machine.dispatch(
            TaskEvent.INSERTION_FINISHED
        )
        machine.dispatch(
            TaskEvent.VERIFY_FAILURE
        )

        self.assertEqual(
            machine.state,
            TaskState.FAILED,
        )

        self.assertEqual(
            machine.attempts,
            2,
        )

    def test_act_failure(self):
        machine = TaskStateMachine()

        machine.dispatch(
            TaskEvent.START
        )

        machine.dispatch(
            TaskEvent.HOME_REACHED
        )

        machine.dispatch(
            TaskEvent.ACT_APPROACH_FAILED
        )

        self.assertEqual(
            machine.state,
            TaskState.FAILED,
        )

    def test_abort(self):
        machine = TaskStateMachine()

        machine.dispatch(
            TaskEvent.START
        )

        machine.dispatch(
            TaskEvent.ABORT
        )

        self.assertEqual(
            machine.state,
            TaskState.FAILED,
        )


if __name__ == "__main__":
    unittest.main()