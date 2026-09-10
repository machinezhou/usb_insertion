from __future__ import annotations

from usb_insertion.core.datatypes import Observation
from usb_insertion.core.protocols import (
    CameraBackend,
    RobotBackend,
)


class FakeObservationSource:
    """
    将 FakeRobot 和 FakeCameras 组合成完整 Observation。

    这个类主要用于测试。

    正式 SO-101 runtime 不会使用这个实现；
    正式环境会直接适配 LeRobot robot.get_observation()。
    """

    def __init__(
        self,
        robot: RobotBackend,
        cameras: CameraBackend,
    ):
        self._robot = robot
        self._cameras = cameras

    def get_observation(self) -> Observation:
        robot_state = self._robot.get_state()
        camera_frames = self._cameras.read()

        return Observation(
            robot=robot_state,
            cameras=camera_frames,
        )