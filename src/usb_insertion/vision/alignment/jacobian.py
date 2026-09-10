from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ImageJacobianModel:
    """
    Visual Servo 使用的局部 Image Jacobian。

    matrix:

        shape = (
            number_of_visual_errors,
            number_of_controlled_joints,
        )

    局部近似关系：

        de ≈ J @ dq

    其中：

        e  = image-space visual error
        dq = joint-space increment
    """

    matrix: np.ndarray

    error_keys: tuple[str, ...]

    controlled_joint_indices: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape


def _validate_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Image Jacobian must be a 2-D matrix"
        )

    if (
        matrix.shape[0] <= 0
        or matrix.shape[1] <= 0
    ):
        raise ValueError(
            "Image Jacobian must not be empty"
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "Image Jacobian contains NaN or Inf"
        )

    return matrix


def save_image_jacobian(
    path: str | Path,
    matrix: np.ndarray,
) -> Path:
    """
    保存 Jacobian。

    当前项目标准格式：

        .npy

    原因：
        configs/vision.yaml 已经保存了
        error_keys / controlled_joint_indices，
        不需要把这些 metadata 再塞进 npz。
    """

    path = Path(path)

    if path.suffix != ".npy":
        raise ValueError(
            "Image Jacobian path must end with .npy"
        )

    matrix = _validate_matrix(
        matrix
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        matrix,
        allow_pickle=False,
    )

    return path


def load_image_jacobian(
    path: str | Path,
    *,
    error_keys: Sequence[str] | None = None,
    controlled_joint_indices: Sequence[int] | None = None,
) -> ImageJacobianModel:
    """
    加载 Image Jacobian。

    主格式：
        .npy

    同时兼容旧的 .npz 文件，避免开发过程中
    已产生的标定文件完全失效。

    注意：

        np.load("xxx.npy")
            -> numpy.ndarray

        np.load("xxx.npz")
            -> numpy.lib.npyio.NpzFile

    因此绝对不能无条件使用：

        with np.load(...) as data
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image Jacobian not found: {path}"
        )

    loaded = np.load(
        path,
        allow_pickle=False,
    )

    file_error_keys: tuple[str, ...] | None = None

    file_joint_indices: tuple[int, ...] | None = None

    # ---------------------------------------------------------
    # Standard .npy format
    # ---------------------------------------------------------

    if isinstance(
        loaded,
        np.ndarray,
    ):
        matrix = loaded

    # ---------------------------------------------------------
    # Backward-compatible .npz format
    # ---------------------------------------------------------

    elif isinstance(
        loaded,
        np.lib.npyio.NpzFile,
    ):
        try:
            if "jacobian" in loaded.files:
                matrix = loaded[
                    "jacobian"
                ]

            elif "matrix" in loaded.files:
                matrix = loaded[
                    "matrix"
                ]

            else:
                raise ValueError(
                    "NPZ Jacobian file must contain "
                    "'jacobian' or 'matrix'"
                )

            if "error_keys" in loaded.files:
                file_error_keys = tuple(
                    str(value)
                    for value in loaded[
                        "error_keys"
                    ].tolist()
                )

            if (
                "controlled_joint_indices"
                in loaded.files
            ):
                file_joint_indices = tuple(
                    int(value)
                    for value in loaded[
                        "controlled_joint_indices"
                    ].tolist()
                )

        finally:
            loaded.close()

    else:
        raise TypeError(
            "Unsupported np.load result type: "
            f"{type(loaded).__name__}"
        )

    matrix = _validate_matrix(
        matrix
    )

    # ---------------------------------------------------------
    # Metadata primarily comes from configs/vision.yaml.
    #
    # Old NPZ metadata is only fallback.
    # ---------------------------------------------------------

    if error_keys is None:
        error_keys_tuple = (
            file_error_keys
            if file_error_keys is not None
            else tuple()
        )

    else:
        error_keys_tuple = tuple(
            str(value)
            for value in error_keys
        )

    if controlled_joint_indices is None:
        joint_indices_tuple = (
            file_joint_indices
            if file_joint_indices is not None
            else tuple()
        )

    else:
        joint_indices_tuple = tuple(
            int(value)
            for value in controlled_joint_indices
        )

    # ---------------------------------------------------------
    # Validate matrix against metadata when available.
    # ---------------------------------------------------------

    if (
        error_keys_tuple
        and matrix.shape[0]
        != len(error_keys_tuple)
    ):
        raise ValueError(
            "Image Jacobian row count does not "
            "match error_keys: "
            f"{matrix.shape[0]} != "
            f"{len(error_keys_tuple)}"
        )

    if (
        joint_indices_tuple
        and matrix.shape[1]
        != len(joint_indices_tuple)
    ):
        raise ValueError(
            "Image Jacobian column count does not "
            "match controlled_joint_indices: "
            f"{matrix.shape[1]} != "
            f"{len(joint_indices_tuple)}"
        )

    return ImageJacobianModel(
        matrix=matrix,
        error_keys=error_keys_tuple,
        controlled_joint_indices=(
            joint_indices_tuple
        ),
    )
