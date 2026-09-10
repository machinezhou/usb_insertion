from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from usb_insertion.core.datatypes import FrameBundle


class FakeCameras:
    """
    内存中的假多相机后端。

    用于：
    - 测试 Observation 数据流
    - 测试业务流程
    - 测试 Fake Policy
    - 测试 Success Detector
    - dry-run

    不连接任何真实摄像头。

    默认模拟当前项目中的三路相机：
    - top
    - wrist
    - side
    """

    def __init__(
        self,
        camera_names: Sequence[str] = (
            "top",
            "wrist",
            "side",
        ),
        width: int = 640,
        height: int = 480,
        channels: int = 3,
    ):
        if not camera_names:
            raise ValueError(
                "camera_names must not be empty"
            )

        if len(set(camera_names)) != len(camera_names):
            raise ValueError(
                "camera_names must be unique"
            )

        if width <= 0:
            raise ValueError(
                "width must be greater than 0"
            )

        if height <= 0:
            raise ValueError(
                "height must be greater than 0"
            )

        if channels <= 0:
            raise ValueError(
                "channels must be greater than 0"
            )

        self._camera_names = tuple(camera_names)
        self._width = width
        self._height = height
        self._channels = channels

        self._opened = False

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._camera_names

    def open(self) -> None:
        """
        模拟打开所有相机。
        """
        self._opened = True

    def close(self) -> None:
        """
        模拟关闭所有相机。
        """
        self._opened = False

    def read(self) -> FrameBundle:
        """
        返回一组假的多相机图像。

        每个 camera frame 都是：

            shape = (height, width, channels)
            dtype = uint8

        当前使用全黑图像。
        """

        self._require_opened()

        frames = {
            name: np.zeros(
                (
                    self._height,
                    self._width,
                    self._channels,
                ),
                dtype=np.uint8,
            )
            for name in self._camera_names
        }

        return FrameBundle(
            frames=frames,
            timestamp_s=time.monotonic(),
        )

    def _require_opened(self) -> None:
        if not self._opened:
            raise RuntimeError(
                "FakeCameras are not open"
            )