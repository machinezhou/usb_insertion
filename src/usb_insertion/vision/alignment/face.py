from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from usb_insertion.core.datatypes import (
    AlignmentEstimate,
    Observation,
)
from usb_insertion.core.protocols import (
    AlignmentEstimator,
)
from usb_insertion.vision.roi import ImageROI


@dataclass(frozen=True, slots=True)
class PlugFaceConfig:
    """
    USB 正反面分类配置。
    """

    camera_name: str
    roi: ImageROI

    templates: dict[str, Path]

    correct_label: str

    min_good_matches: int = 6
    ratio_test: float = 0.75
    min_score_margin: float = 0.15


@dataclass(frozen=True, slots=True)
class PlugFaceResult:
    visible: bool

    label: str | None
    confidence: float

    scores: dict[str, float]

    reason: str = ""


class TemplatePlugFaceClassifier:
    """
    USB 正反面分类器 V1。

    使用你的 USB 自身明显标识。

    优先 SIFT；
    不支持 SIFT 时使用 ORB。
    """

    def __init__(
        self,
        config: PlugFaceConfig,
    ):
        self.config = config

        if len(config.templates) < 2:
            raise ValueError(
                "Plug-face classifier needs "
                "at least two templates"
            )

        if (
            config.correct_label
            not in config.templates
        ):
            raise ValueError(
                "correct_label is not "
                "present in templates"
            )

        if config.min_good_matches < 2:
            raise ValueError(
                "min_good_matches must be >= 2"
            )

        if not (
            0.0
            < config.ratio_test
            < 1.0
        ):
            raise ValueError(
                "ratio_test must be in (0, 1)"
            )

        if config.min_score_margin < 0:
            raise ValueError(
                "min_score_margin must be >= 0"
            )

        if hasattr(
            cv2,
            "SIFT_create",
        ):
            self._detector = (
                cv2.SIFT_create(
                    nfeatures=1200
                )
            )

            self._matcher = (
                cv2.BFMatcher(
                    cv2.NORM_L2
                )
            )

            self._feature_type = "SIFT"

        else:
            self._detector = (
                cv2.ORB_create(
                    nfeatures=1800
                )
            )

            self._matcher = (
                cv2.BFMatcher(
                    cv2.NORM_HAMMING
                )
            )

            self._feature_type = "ORB"

        self._template_descriptors = {}
        self._template_keypoint_counts = {}

        for label, path in (
            config.templates.items()
        ):
            image = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE,
            )

            if image is None:
                raise FileNotFoundError(
                    "Unable to load plug-face "
                    f"template '{label}': {path}"
                )

            (
                keypoints,
                descriptors,
            ) = (
                self._detector
                .detectAndCompute(
                    image,
                    None,
                )
            )

            if (
                descriptors is None
                or len(keypoints)
                < config.min_good_matches
            ):
                raise ValueError(
                    f"Plug-face template "
                    f"'{label}' has too few "
                    "visual features"
                )

            self._template_descriptors[
                label
            ] = descriptors

            self._template_keypoint_counts[
                label
            ] = len(keypoints)

    @property
    def feature_type(self) -> str:
        return self._feature_type

    def classify(
        self,
        frame_rgb: np.ndarray,
    ) -> PlugFaceResult:

        crop = self.config.roi.crop(
            frame_rgb,
            copy=False,
        )

        if (
            crop.ndim != 3
            or crop.shape[2] != 3
        ):
            return PlugFaceResult(
                visible=False,
                label=None,
                confidence=0.0,
                scores={},
                reason="invalid_frame_shape",
            )

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2GRAY,
        )

        (
            keypoints,
            descriptors,
        ) = (
            self._detector
            .detectAndCompute(
                gray,
                None,
            )
        )

        if (
            descriptors is None
            or len(keypoints)
            < self.config
            .min_good_matches
        ):
            return PlugFaceResult(
                visible=False,
                label=None,
                confidence=0.0,
                scores={},
                reason=(
                    "not_enough_current_features"
                ),
            )

        scores = {}
        good_counts = {}

        for (
            label,
            template_descriptors,
        ) in (
            self._template_descriptors
            .items()
        ):
            pairs = (
                self._matcher.knnMatch(
                    template_descriptors,
                    descriptors,
                    k=2,
                )
            )

            good = 0

            for pair in pairs:
                if len(pair) != 2:
                    continue

                first, second = pair

                if (
                    first.distance
                    < self.config.ratio_test
                    * second.distance
                ):
                    good += 1

            good_counts[label] = good

            denominator = max(
                self.config
                .min_good_matches,
                self
                ._template_keypoint_counts[
                    label
                ],
            )

            scores[label] = (
                good / denominator
            )

        ordered = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        (
            best_label,
            best_score,
        ) = ordered[0]

        second_score = ordered[1][1]

        if (
            good_counts[best_label]
            < self.config
            .min_good_matches
        ):
            return PlugFaceResult(
                visible=False,
                label=None,
                confidence=float(
                    best_score
                ),
                scores=scores,
                reason=(
                    "not_enough_good_matches"
                ),
            )

        margin = (
            best_score
            - second_score
        )

        if (
            margin
            < self.config
            .min_score_margin
        ):
            return PlugFaceResult(
                visible=True,
                label=None,
                confidence=float(
                    max(
                        0.0,
                        margin,
                    )
                ),
                scores=scores,
                reason="ambiguous_face",
            )

        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    margin
                    / max(
                        best_score,
                        1e-12,
                    ),
                ),
            )
        )

        return PlugFaceResult(
            visible=True,
            label=best_label,
            confidence=confidence,
            scores=scores,
            reason="classified",
        )


def load_plug_face_config(
    path: str | Path,
) -> PlugFaceConfig:

    path = Path(path)

    project_root = (
        path.resolve()
        .parents[1]
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(
            file
        )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            "plug-face config root "
            "must be a mapping"
        )

    roi_raw = raw.get("roi")
    templates_raw = raw.get(
        "templates"
    )

    if not isinstance(
        roi_raw,
        dict,
    ):
        raise ValueError(
            "'roi' must be a mapping"
        )

    if not isinstance(
        templates_raw,
        dict,
    ):
        raise ValueError(
            "'templates' must be "
            "a mapping"
        )

    templates = {}

    for label, value in (
        templates_raw.items()
    ):
        template_path = Path(
            str(value)
        )

        if not (
            template_path.is_absolute()
        ):
            template_path = (
                project_root
                / template_path
            )

        templates[
            str(label)
        ] = template_path

    return PlugFaceConfig(
        camera_name=str(
            raw.get(
                "camera_name",
                "wrist",
            )
        ),
        roi=ImageROI(
            x=int(
                roi_raw["x"]
            ),
            y=int(
                roi_raw["y"]
            ),
            width=int(
                roi_raw["width"]
            ),
            height=int(
                roi_raw["height"]
            ),
        ),
        templates=templates,
        correct_label=str(
            raw.get(
                "correct_label",
                "marked",
            )
        ),
        min_good_matches=int(
            raw.get(
                "min_good_matches",
                6,
            )
        ),
        ratio_test=float(
            raw.get(
                "ratio_test",
                0.75,
            )
        ),
        min_score_margin=float(
            raw.get(
                "min_score_margin",
                0.15,
            )
        ),
    )


class FaceAwareAlignmentEstimator:
    """
    在几何 AlignmentEstimator 外增加
    USB 正反面安全 gate。

    错面或者无法可靠分类：
        不允许交给 Visual Servo。
    """

    def __init__(
        self,
        geometric_estimator: AlignmentEstimator,
        face_classifier: (
            TemplatePlugFaceClassifier
        ),
    ):
        self._geometric_estimator = (
            geometric_estimator
        )

        self._face_classifier = (
            face_classifier
        )

    def reset(self) -> None:
        self._geometric_estimator.reset()

    def estimate(
        self,
        observation: Observation,
    ) -> AlignmentEstimate:

        geometric = (
            self._geometric_estimator
            .estimate(
                observation
            )
        )

        if not geometric.visible:
            return geometric

        camera_name = (
            self._face_classifier
            .config
            .camera_name
        )

        frames = (
            observation
            .cameras
            .frames
        )

        if camera_name not in frames:
            return AlignmentEstimate(
                visible=False,
                in_capture_region=False,
                aligned=False,
                errors=geometric.errors,
                confidence=0.0,
                plug_face=None,
                reason=(
                    "missing_face_camera:"
                    f"{camera_name}"
                ),
            )

        face = (
            self._face_classifier
            .classify(
                frames[
                    camera_name
                ]
            )
        )

        correct = (
            face.visible
            and face.label
            == self._face_classifier
            .config
            .correct_label
        )

        if not correct:
            return AlignmentEstimate(
                visible=False,
                in_capture_region=False,
                aligned=False,
                errors=geometric.errors,
                confidence=min(
                    geometric.confidence,
                    face.confidence,
                ),
                plug_face=face.label,
                reason=(
                    f"plug_face:"
                    f"{face.reason}:"
                    f"{face.label}"
                ),
            )

        return AlignmentEstimate(
            visible=True,
            in_capture_region=(
                geometric
                .in_capture_region
            ),
            aligned=(
                geometric.aligned
            ),
            errors=geometric.errors,
            confidence=min(
                geometric.confidence,
                face.confidence,
            ),
            plug_face=face.label,
            reason=geometric.reason,
        )
