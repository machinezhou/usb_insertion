from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ImageROI:
    """
    图像矩形 ROI。

    坐标约定：

        x: 左上角横坐标
        y: 左上角纵坐标
        width: ROI 宽度
        height: ROI 高度

    对 ndarray 图像：

        frame[y:y+height, x:x+width]
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0:
            raise ValueError(
                "ROI x must be >= 0"
            )

        if self.y < 0:
            raise ValueError(
                "ROI y must be >= 0"
            )

        if self.width <= 0:
            raise ValueError(
                "ROI width must be > 0"
            )

        if self.height <= 0:
            raise ValueError(
                "ROI height must be > 0"
            )

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def validate_frame(
        self,
        frame: np.ndarray,
    ) -> None:
        """
        检查 ROI 是否落在 frame 内。
        """

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "frame must be numpy.ndarray"
            )

        if frame.ndim not in (2, 3):
            raise ValueError(
                "frame must be HxW or HxWxC"
            )

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        if self.x2 > frame_width:
            raise ValueError(
                "ROI exceeds frame width: "
                f"x2={self.x2}, "
                f"frame_width={frame_width}"
            )

        if self.y2 > frame_height:
            raise ValueError(
                "ROI exceeds frame height: "
                f"y2={self.y2}, "
                f"frame_height={frame_height}"
            )

    def crop(
        self,
        frame: np.ndarray,
        *,
        copy: bool = True,
    ) -> np.ndarray:
        """
        从 frame 中裁剪 ROI。

        copy=True:
            返回独立 ndarray。

        copy=False:
            返回原图 view。
        """

        self.validate_frame(
            frame
        )

        cropped = frame[
            self.y:self.y2,
            self.x:self.x2,
        ]

        if copy:
            return cropped.copy()

        return cropped