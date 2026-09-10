import json
import tempfile
import unittest
from pathlib import Path

from usb_insertion.cli.dry_run import (
    run_dry_run,
)
from usb_insertion.workflow.states import (
    TaskState,
)


class TestDryRun(unittest.TestCase):
    def test_complete_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "dry_run.jsonl"
            )

            result, returned_path = (
                run_dry_run(
                    log_path=path,
                    verbose=False,
                )
            )

            self.assertTrue(
                result.success
            )

            self.assertEqual(
                result.final_state,
                TaskState.DONE,
            )

            self.assertEqual(
                result.attempts,
                2,
            )

            self.assertEqual(
                returned_path,
                path,
            )

            self.assertTrue(
                path.exists()
            )

    def test_act_is_used_before_visual_servo(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "dry_run.jsonl"
            )

            result, _ = run_dry_run(
                log_path=path,
                verbose=False,
            )

            self.assertGreater(
                result.act_policy_steps,
                0,
            )

            self.assertGreater(
                result.alignment_steps,
                0,
            )

            self.assertGreater(
                result.insertion_steps,
                0,
            )

    def test_expected_log_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "dry_run.jsonl"
            )

            run_dry_run(
                log_path=path,
                verbose=False,
            )

            records = [
                json.loads(line)
                for line
                in path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

            events = {
                record["event"]
                for record in records
            }

            self.assertIn(
                "act_control_step",
                events,
            )

            self.assertIn(
                "visual_alignment_step",
                events,
            )

            self.assertIn(
                "visual_insertion_step",
                events,
            )

            self.assertIn(
                "verification",
                events,
            )

            self.assertIn(
                "task_finished",
                events,
            )


if __name__ == "__main__":
    unittest.main()