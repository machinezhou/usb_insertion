import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from usb_insertion.vision.alignment.face import (
    PlugFaceConfig,
    TemplatePlugFaceClassifier,
)
from usb_insertion.vision.roi import (
    ImageROI,
)


class TestPlugFace(
    unittest.TestCase
):
    def test_exact_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            rng = (
                np.random
                .default_rng(7)
            )

            marked = rng.integers(
                0,
                256,
                size=(120, 160),
                dtype=np.uint8,
            )

            unmarked = rng.integers(
                0,
                256,
                size=(120, 160),
                dtype=np.uint8,
            )

            cv2.imwrite(
                str(
                    root
                    / "marked.png"
                ),
                marked,
            )

            cv2.imwrite(
                str(
                    root
                    / "unmarked.png"
                ),
                unmarked,
            )

            config = PlugFaceConfig(
                camera_name="wrist",
                roi=ImageROI(
                    x=20,
                    y=10,
                    width=160,
                    height=120,
                ),
                templates={
                    "marked": (
                        root
                        / "marked.png"
                    ),
                    "unmarked": (
                        root
                        / "unmarked.png"
                    ),
                },
                correct_label=(
                    "marked"
                ),
                min_score_margin=0.05,
            )

            classifier = (
                TemplatePlugFaceClassifier(
                    config
                )
            )

            frame = np.zeros(
                (
                    160,
                    220,
                    3,
                ),
                dtype=np.uint8,
            )

            frame[
                10:130,
                20:180,
            ] = cv2.cvtColor(
                marked,
                cv2.COLOR_GRAY2RGB,
            )

            result = (
                classifier.classify(
                    frame
                )
            )

            self.assertTrue(
                result.visible
            )

            self.assertEqual(
                result.label,
                "marked",
            )


if __name__ == "__main__":
    unittest.main()
