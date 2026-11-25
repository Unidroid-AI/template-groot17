# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map tabletop observations to a GR00T N1.7 policy and back to absolute joints."""

from __future__ import annotations

import numpy as np

IMAGE_HW = (180, 320)


def resize_hwc(images: np.ndarray, height: int = IMAGE_HW[0], width: int = IMAGE_HW[1]) -> np.ndarray:
    """Nearest-neighbor resize of uint8 HWC images.

    Args:
        images: Shape [H, W, C] or [N, H, W, C].
    """
    array = np.asarray(images)
    batched = array.ndim == 4
    if not batched:
        array = array[None]
    count, src_h, src_w, channels = array.shape
    row = np.clip((np.arange(height) * src_h / height).astype(np.int64), 0, src_h - 1)
    col = np.clip((np.arange(width) * src_w / width).astype(np.int64), 0, src_w - 1)
    resized = array[:, row][:, :, col]
    assert resized.shape == (count, height, width, channels)
    return resized if batched else resized[0]


def joint_action_chunk(raw, horizon: int) -> np.ndarray:
    """Take GR00T's decoded joint targets [rad] and gripper as an 8-D command chunk.

    ``Gr00tPolicy.get_action`` already adds the checkpoint's relative joint head to the
    state it was given, so ``joint_position`` is absolute. Adding the observed joints again
    doubles the pose and drives the arm into its joint limits.

    Args:
        raw: The policy result, an action dict or an ``(action, info)`` pair.
        horizon: Number of steps to keep.

    Returns:
        Actions, shape [N, H, 8]. The last column is the gripper, 0 open and 1 closed.
    """
    action = raw[0] if isinstance(raw, tuple) else raw
    action = action.get("action", action)
    joints = np.asarray(_action_value(action, "joint_position"), dtype=np.float32)[:, :horizon]
    gripper = np.asarray(_action_value(action, "gripper_position"), dtype=np.float32)[:, :horizon]
    if gripper.ndim == 2:
        gripper = gripper[..., None]
    return np.concatenate((joints, gripper), axis=-1).astype(np.float32)


def _action_value(action: dict, name: str):
    prefixed = f"action.{name}"
    if prefixed in action:
        return action[prefixed]
    if name in action:
        return action[name]
    raise KeyError(f"GR00T action has no {name!r}. Keys: {sorted(action)}")
