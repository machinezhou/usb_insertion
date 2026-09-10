from __future__ import annotations


class ConsecutiveTrue:
    """
    连续 True 确认器。

    只有连续 required 次输入 True，
    最终状态才变成 True。

    任何一次 False 都会把计数清零。
    """

    def __init__(
        self,
        required: int,
    ):
        if required <= 0:
            raise ValueError(
                "required must be greater than 0"
            )

        self._required = required
        self._count = 0

    @property
    def required(self) -> int:
        return self._required

    @property
    def count(self) -> int:
        return self._count

    @property
    def confirmed(self) -> bool:
        return (
            self._count
            >= self._required
        )

    def reset(self) -> None:
        self._count = 0

    def update(
        self,
        value: bool,
    ) -> bool:
        if value:
            self._count += 1
        else:
            self._count = 0

        return self.confirmed