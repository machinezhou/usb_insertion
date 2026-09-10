import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from usb_insertion.core.datatypes import Action
from usb_insertion.runtime.jsonl_logger import (
    JsonlLogger,
    to_jsonable,
)
from usb_insertion.workflow.states import (
    TaskState,
)


class TestJsonlLogger(unittest.TestCase):
    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp)
                / "nested"
                / "run.jsonl"
            )

            logger = JsonlLogger(path)

            self.assertTrue(
                path.parent.exists()
            )

            self.assertEqual(
                logger.path,
                path,
            )

    def test_write_one_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"

            logger = JsonlLogger(path)

            logger.write(
                "test_event",
                value=123,
            )

            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(
                len(lines),
                1,
            )

            record = json.loads(
                lines[0]
            )

            self.assertEqual(
                record["event"],
                "test_event",
            )

            self.assertEqual(
                record["value"],
                123,
            )

            self.assertIn(
                "wall_time_s",
                record,
            )

            self.assertIn(
                "monotonic_s",
                record,
            )

    def test_append_multiple_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"

            logger = JsonlLogger(path)

            logger.write(
                "event_1"
            )

            logger.write(
                "event_2"
            )

            lines = path.read_text(
                encoding="utf-8"
            ).splitlines()

            self.assertEqual(
                len(lines),
                2,
            )

            first = json.loads(
                lines[0]
            )

            second = json.loads(
                lines[1]
            )

            self.assertEqual(
                first["event"],
                "event_1",
            )

            self.assertEqual(
                second["event"],
                "event_2",
            )

    def test_numpy_array_serialization(self):
        value = np.array([
            1.0,
            2.0,
            3.0,
        ])

        result = to_jsonable(
            value
        )

        self.assertEqual(
            result,
            [
                1.0,
                2.0,
                3.0,
            ],
        )

    def test_numpy_scalar_serialization(self):
        value = np.float64(1.25)

        result = to_jsonable(
            value
        )

        self.assertEqual(
            result,
            1.25,
        )

    def test_dataclass_serialization(self):
        action = Action(
            q=np.array([
                0.1,
                0.2,
                0.3,
            ]),
            timestamp_s=1.0,
        )

        result = to_jsonable(
            action
        )

        self.assertEqual(
            result["q"],
            [
                0.1,
                0.2,
                0.3,
            ],
        )

        self.assertEqual(
            result["timestamp_s"],
            1.0,
        )

    def test_enum_serialization(self):
        result = to_jsonable(
            TaskState.ACT_PICK_APPROACH
        )

        self.assertEqual(
            result,
            "ACT_PICK_APPROACH",
        )

    def test_empty_event_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"

            logger = JsonlLogger(path)

            with self.assertRaises(
                ValueError
            ):
                logger.write("")


if __name__ == "__main__":
    unittest.main()