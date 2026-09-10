from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """
    将项目中常见的数据转换成 JSON 可序列化形式。

    支持：
    - Python 基本类型
    - numpy.ndarray
    - numpy scalar
    - dataclass
    - Enum
    - Path
    - dict
    - list / tuple
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Enum):
        return value.name

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return to_jsonable(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item)
            for item in value
        ]

    raise TypeError(
        "Unsupported JSON log value type: "
        f"{type(value).__name__}"
    )


class JsonlLogger:
    """
    简单的 JSON Lines 结构化日志器。

    每调用一次 write()，写入一行独立 JSON。

    示例：

        {
            "wall_time_s": 1234567890.0,
            "monotonic_s": 123.4,
            "event": "state_transition",
            "from_state": "PREINSERT",
            "to_state": "ACT_ATTEMPT"
        }
    """

    def __init__(
        self,
        path: str | Path,
    ):
        self._path = Path(path)

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def path(self) -> Path:
        return self._path

    def write(
        self,
        event: str,
        **payload: Any,
    ) -> None:
        """
        向 JSONL 文件追加一条事件。
        """

        if not isinstance(event, str):
            raise TypeError(
                "event must be a string"
            )

        if not event.strip():
            raise ValueError(
                "event must not be empty"
            )

        record = {
            "wall_time_s": time.time(),
            "monotonic_s": time.monotonic(),
            "event": event,
        }

        record.update(
            {
                key: to_jsonable(value)
                for key, value in payload.items()
            }
        )

        with self._path.open(
            "a",
            encoding="utf-8",
        ) as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            file.write("\n")