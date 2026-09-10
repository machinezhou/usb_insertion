from __future__ import annotations

import numpy as np


def quintic_smoothstep(u: np.ndarray) -> np.ndarray:
    """
    五次 smoothstep:

        s(u) = 10u^3 - 15u^4 + 6u^5

    u 的有效范围为 [0, 1]。

    在起点和终点：
    - 速度为 0
    - 加速度为 0
    """
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def joint_trajectory(
    q_start: np.ndarray,
    q_target: np.ndarray,
    duration_s: float,
    fps: float,
) -> np.ndarray:
    """
    生成从 q_start 到 q_target 的平滑关节轨迹。

    Parameters
    ----------
    q_start:
        起始关节位置，shape = (dof,)

    q_target:
        目标关节位置，shape = (dof,)

    duration_s:
        运动持续时间，单位秒。

    fps:
        控制频率，例如 30 Hz。

    Returns
    -------
    np.ndarray
        shape = (num_steps, dof)

        第一行为 q_start。
        最后一行为 q_target。
    """

    q_start = np.asarray(q_start, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)

    if q_start.ndim != 1:
        raise ValueError("q_start must be a 1-D array")

    if q_target.ndim != 1:
        raise ValueError("q_target must be a 1-D array")

    if q_start.shape != q_target.shape:
        raise ValueError(
            "q_start and q_target must have the same shape: "
            f"{q_start.shape} != {q_target.shape}"
        )

    if duration_s <= 0:
        raise ValueError("duration_s must be greater than 0")

    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    num_steps = max(
        2,
        int(round(duration_s * fps)) + 1,
    )

    u = np.linspace(
        0.0,
        1.0,
        num_steps,
        dtype=np.float64,
    )

    s = quintic_smoothstep(u)

    delta_q = q_target - q_start

    trajectory = (
        q_start[None, :]
        + s[:, None] * delta_q[None, :]
    )

    # 明确保证两个端点精确等于输入。
    trajectory[0] = q_start
    trajectory[-1] = q_target

    return trajectory