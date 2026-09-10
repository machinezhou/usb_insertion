from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from usb_insertion.vision.alignment.config import (
    CameraAlignmentConfig,
)


def wrap_angle_deg(
    angle_deg: float,
) -> float:
    """
    将角度限制到 [-180, 180)。
    """

    return (
        angle_deg + 180.0
    ) % 360.0 - 180.0


def angle_between_points_deg(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """
    p1 -> p2 的二维角度。
    """

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.degrees(
        math.atan2(
            dy,
            dx,
        )
    )


@dataclass(slots=True)
class CameraPoseObservation:
    visible: bool

    tip_xy: tuple[float, float] | None = None

    axis_xy: tuple[float, float] | None = None

    axis_angle_deg: float | None = None

    good_matches: int = 0
    inliers: int = 0

    confidence: float = 0.0

    reason: str = ""


class TemplatePoseTracker:
    """
    使用局部特征 + Homography
    估计 USB 插头在单个 camera 中的二维 pose。

    优先使用 SIFT；
    如果环境 OpenCV 没有 SIFT，则 fallback 到 ORB。
    """

    def __init__(
        self,
        config: CameraAlignmentConfig,
    ):
        self.config = config

        template_bgr = cv2.imread(
            str(
                config.template_path
            ),
            cv2.IMREAD_COLOR,
        )

        if template_bgr is None:
            raise FileNotFoundError(
                "Unable to load template: "
                f"{config.template_path}"
            )

        self._template_gray = (
            cv2.cvtColor(
                template_bgr,
                cv2.COLOR_BGR2GRAY,
            )
        )

        if hasattr(
            cv2,
            "SIFT_create",
        ):
            self._detector = (
                cv2.SIFT_create(
                    nfeatures=1000
                )
            )

            self._matcher = (
                cv2.BFMatcher(
                    cv2.NORM_L2,
                )
            )

            self._feature_type = (
                "SIFT"
            )

        else:
            self._detector = (
                cv2.ORB_create(
                    nfeatures=1500
                )
            )

            self._matcher = (
                cv2.BFMatcher(
                    cv2.NORM_HAMMING,
                )
            )

            self._feature_type = (
                "ORB"
            )

        (
            self._template_keypoints,
            self._template_descriptors,
        ) = self._detector.detectAndCompute(
            self._template_gray,
            None,
        )

        if (
            self._template_descriptors
            is None
            or len(
                self._template_keypoints
            ) < config.min_matches
        ):
            raise ValueError(
                f"Template for camera "
                f"'{config.camera_name}' "
                "does not contain enough "
                "visual features"
            )

    @property
    def feature_type(self) -> str:
        return self._feature_type

    @staticmethod
    def _transform_point(
        point_xy: tuple[float, float],
        homography: np.ndarray,
    ) -> tuple[float, float]:

        point = np.array(
            [
                [
                    point_xy[
                        0
                    ],
                    point_xy[
                        1
                    ],
                ]
            ],
            dtype=np.float32,
        ).reshape(
            1,
            1,
            2,
        )

        transformed = (
            cv2.perspectiveTransform(
                point,
                homography,
            )
        )

        return (
            float(
                transformed[
                    0,
                    0,
                    0,
                ]
            ),
            float(
                transformed[
                    0,
                    0,
                    1,
                ]
            ),
        )

    def detect(
        self,
        frame_rgb: np.ndarray,
    ) -> CameraPoseObservation:

        roi = self.config.roi

        crop_rgb = roi.crop(
            frame_rgb,
            copy=False,
        )

        if (
            crop_rgb.ndim != 3
            or crop_rgb.shape[2] != 3
        ):
            return CameraPoseObservation(
                visible=False,
                reason=(
                    "invalid_frame_shape"
                ),
            )

        crop_gray = cv2.cvtColor(
            crop_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        (
            current_keypoints,
            current_descriptors,
        ) = self._detector.detectAndCompute(
            crop_gray,
            None,
        )

        if (
            current_descriptors
            is None
            or len(current_keypoints)
            < self.config.min_matches
        ):
            return CameraPoseObservation(
                visible=False,
                reason=(
                    "not_enough_current_features"
                ),
            )

        matches = (
            self._matcher.knnMatch(
                self._template_descriptors,
                current_descriptors,
                k=2,
            )
        )

        good = []

        for pair in matches:
            if len(pair) != 2:
                continue

            first, second = pair

            if (
                first.distance
                < self.config.ratio_test
                * second.distance
            ):
                good.append(
                    first
                )

        if len(good) < (
            self.config.min_matches
        ):
            return CameraPoseObservation(
                visible=False,
                good_matches=len(good),
                reason="not_enough_matches",
            )

        src_points = np.float32(
            [
                self._template_keypoints[
                    match.queryIdx
                ].pt
                for match in good
            ]
        ).reshape(
            -1,
            1,
            2,
        )

        dst_points = np.float32(
            [
                current_keypoints[
                    match.trainIdx
                ].pt
                for match in good
            ]
        ).reshape(
            -1,
            1,
            2,
        )

        homography, mask = (
            cv2.findHomography(
                src_points,
                dst_points,
                cv2.RANSAC,
                self.config
                .ransac_reproj_threshold,
            )
        )

        if (
            homography is None
            or mask is None
        ):
            return CameraPoseObservation(
                visible=False,
                good_matches=len(good),
                reason="homography_failed",
            )

        inliers = int(
            mask.ravel().sum()
        )

        if inliers < 4:
            return CameraPoseObservation(
                visible=False,
                good_matches=len(good),
                inliers=inliers,
                reason="not_enough_inliers",
            )

        tip_local = (
            self._transform_point(
                self.config
                .template_tip_xy,
                homography,
            )
        )

        axis_local = (
            self._transform_point(
                self.config
                .template_axis_xy,
                homography,
            )
        )

        tip_full = (
            tip_local[0]
            + roi.x,
            tip_local[1]
            + roi.y,
        )

        axis_full = (
            axis_local[0]
            + roi.x,
            axis_local[1]
            + roi.y,
        )

        axis_angle = (
            angle_between_points_deg(
                tip_full,
                axis_full,
            )
        )

        confidence = (
            inliers
            / max(
                len(good),
                1,
            )
        )

        return CameraPoseObservation(
            visible=True,
            tip_xy=tip_full,
            axis_xy=axis_full,
            axis_angle_deg=(
                axis_angle
            ),
            good_matches=len(good),
            inliers=inliers,
            confidence=float(
                confidence
            ),
            reason="detected",
        )