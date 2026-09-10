from __future__ import annotations

import math

import cv2
import numpy as np

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
    Observation,
)
from usb_insertion.vision.alignment.config import (
    VisionAlignmentConfig,
)
from usb_insertion.vision.alignment.features import (
    CameraPoseObservation,
    TemplatePoseTracker,
    wrap_angle_deg,
)


class TemplateAlignmentEstimator:
    """
    USB AlignmentEstimator V1。

    假设：
    - 插座固定
    - top camera 固定
    - side camera 固定
    - 已经标定理想 Pre-insert USB pose

    因此运行时只检测 USB pose，
    然后与理想 reference pose 比较。
    """

    def __init__(
        self,
        config: VisionAlignmentConfig,
    ):
        self.config = config

        self._trackers = {
            name: TemplatePoseTracker(
                camera_config
            )
            for name, camera_config
            in config.cameras.items()
        }

        self._last_pose: dict[
            str,
            CameraPoseObservation,
        ] = {}

    def reset(self) -> None:
        self._last_pose = {}

    @property
    def last_pose(
        self,
    ) -> dict[
        str,
        CameraPoseObservation,
    ]:
        return dict(
            self._last_pose
        )

    def estimate(
        self,
        observation: Observation,
    ) -> AlignmentEstimate:

        frames = (
            observation
            .cameras
            .frames
        )

        poses: dict[
            str,
            CameraPoseObservation,
        ] = {}

        errors: dict[
            str,
            float,
        ] = {}

        confidences = []

        for camera_name, tracker in (
            self._trackers.items()
        ):
            if camera_name not in frames:
                return AlignmentEstimate(
                    visible=False,
                    in_capture_region=False,
                    aligned=False,
                    reason=(
                        "missing_camera:"
                        f"{camera_name}"
                    ),
                )

            pose = tracker.detect(
                frames[camera_name]
            )

            poses[
                camera_name
            ] = pose

            if not pose.visible:
                self._last_pose = poses

                return AlignmentEstimate(
                    visible=False,
                    in_capture_region=False,
                    aligned=False,
                    confidence=0.0,
                    reason=(
                        f"{camera_name}:"
                        f"{pose.reason}"
                    ),
                )

            camera_config = (
                self.config
                .cameras[
                    camera_name
                ]
            )

            assert (
                pose.tip_xy
                is not None
            )

            assert (
                pose.axis_angle_deg
                is not None
            )

            dx = (
                pose.tip_xy[0]
                - camera_config
                .target_tip_xy[0]
            )

            dy = (
                pose.tip_xy[1]
                - camera_config
                .target_tip_xy[1]
            )

            angle_error = (
                wrap_angle_deg(
                    pose.axis_angle_deg
                    - camera_config
                    .target_axis_angle_deg
                )
            )

            errors[
                f"{camera_name}_dx_px"
            ] = float(dx)

            errors[
                f"{camera_name}_dy_px"
            ] = float(dy)

            errors[
                f"{camera_name}_angle_deg"
            ] = float(
                angle_error
            )

            confidences.append(
                pose.confidence
            )

        self._last_pose = poses

        in_capture_region = True
        aligned = True

        for camera_name, camera_config in (
            self.config
            .cameras.items()
        ):
            dx = errors[
                f"{camera_name}_dx_px"
            ]

            dy = errors[
                f"{camera_name}_dy_px"
            ]

            angle = errors[
                f"{camera_name}_angle_deg"
            ]

            tip_distance = math.hypot(
                dx,
                dy,
            )

            if (
                tip_distance
                > camera_config
                .capture_tip_distance_px
                or abs(angle)
                > camera_config
                .capture_angle_deg
            ):
                in_capture_region = False

            if (
                abs(dx)
                > camera_config
                .aligned_dx_px
                or abs(dy)
                > camera_config
                .aligned_dy_px
                or abs(angle)
                > camera_config
                .aligned_angle_deg
            ):
                aligned = False

        confidence = min(
            confidences
        )

        return AlignmentEstimate(
            visible=True,
            in_capture_region=(
                in_capture_region
            ),
            aligned=aligned,
            errors=errors,
            confidence=float(
                confidence
            ),
            plug_face=(
                self.config
                .correct_face_label
            ),
            reason=(
                "aligned"
                if aligned
                else (
                    "in_capture_region"
                    if in_capture_region
                    else "visible"
                )
            ),
        )

    def draw_debug(
        self,
        frame_rgb: np.ndarray,
        camera_name: str,
    ) -> np.ndarray:
        """
        生成用于调试显示的 RGB overlay。
        """

        output = (
            frame_rgb.copy()
        )

        config = (
            self.config
            .cameras[
                camera_name
            ]
        )

        roi = config.roi

        cv2.rectangle(
            output,
            (
                roi.x,
                roi.y,
            ),
            (
                roi.x2,
                roi.y2,
            ),
            (
                255,
                255,
                0,
            ),
            2,
        )

        target = (
            int(
                round(
                    config
                    .target_tip_xy[0]
                )
            ),
            int(
                round(
                    config
                    .target_tip_xy[1]
                )
            ),
        )

        cv2.drawMarker(
            output,
            target,
            (
                0,
                255,
                0,
            ),
            cv2.MARKER_CROSS,
            20,
            2,
        )

        pose = self._last_pose.get(
            camera_name
        )

        if (
            pose is not None
            and pose.visible
            and pose.tip_xy is not None
            and pose.axis_xy is not None
        ):
            tip = (
                int(
                    round(
                        pose.tip_xy[0]
                    )
                ),
                int(
                    round(
                        pose.tip_xy[1]
                    )
                ),
            )

            axis = (
                int(
                    round(
                        pose.axis_xy[0]
                    )
                ),
                int(
                    round(
                        pose.axis_xy[1]
                    )
                ),
            )

            cv2.circle(
                output,
                tip,
                6,
                (
                    255,
                    0,
                    0,
                ),
                -1,
            )

            cv2.line(
                output,
                tip,
                axis,
                (
                    255,
                    0,
                    255,
                ),
                3,
            )

        return output