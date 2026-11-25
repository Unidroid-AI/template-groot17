# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.7 DROID brain for the Franka tabletop task.

``weights/`` is the released ``nvidia/GR00T-N1.7-DROID`` checkpoint.
``embodiment_tag`` selects that checkpoint's relative end-effector plus
relative joint head. The policy decodes that head against the state it is given, so
its joint output is already absolute. It is used as this task's 8-D command with the
gripper as 0 open or 1 closed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from adapt import joint_action_chunk, resize_hwc
from brain_comm.brain import Brain


class Gr00tN17Brain(Brain):
    """Load a GR00T N1.7 checkpoint and return absolute joint actions."""

    # The only per-episode state is the instruction list, so workers can share this replica.
    session_safe = True

    def setup(self, device: str, weights_dir: Path, config: dict) -> None:
        self.horizon = int(config["horizon"])
        self.action_dim = int(config["action_dim"])
        self._tag = config.get("embodiment_tag")
        self._policy = _load_policy(weights_dir, self._tag, device)
        self._instructions: list[str] = []

    def begin_episode(self, episode_ids: list[str], seeds: list[int], instructions: list[str]) -> None:
        self._instructions = list(instructions)

    def end_episode(self, episode_ids: list[str]) -> None:
        """Drop the instruction batch and return unused CUDA blocks to the driver.

        The weights stay loaded. The caching allocator otherwise keeps every
        task's peak, so the next scene asks for memory the card still has free.
        """
        self._instructions = []
        _release_cached_gpu_memory()

    def act(self, obs: dict[str, np.ndarray], step: int) -> np.ndarray:
        joints = np.asarray(obs["joint_pos"], dtype=np.float32)
        grip = np.asarray(obs["gripper_pos"], dtype=np.float32)
        pose = np.asarray(obs["eef_pose"], dtype=np.float32)
        model_obs = {
            "video": {
                "exterior_image_1_left": _video(resize_hwc(obs["cam_exterior_left"])),
                "wrist_image_left": _video(resize_hwc(obs["cam_wrist"])),
            },
            "state": {
                "eef_9d": _time(_eef_9d(pose)),
                "gripper_position": _time(grip),
                "joint_position": _time(joints),
            },
            "language": _language(self._instructions),
        }
        return joint_action_chunk(_infer(self._policy, model_obs), self.horizon)


def _load_policy(weights_dir: Path, embodiment_tag: str | None, device: str):
    if not (weights_dir / "config.json").is_file():
        raise FileNotFoundError(f"GR00T checkpoint is missing config.json in {weights_dir}")
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    _install_rot6d_guard()
    policy = Gr00tPolicy(model_path=str(weights_dir), embodiment_tag=embodiment_tag, device=device)
    _freeze(policy)
    return policy


def _freeze(policy) -> None:
    """Eval mode, no autograd graph. A compiled graph is not kept.

    ``torch.compile`` specializes on the instruction length. Each tabletop task
    is a new length, and every specialization stays resident for the life of
    the process. Eager inference has one working set.
    """
    model = getattr(policy, "model", None)
    if model is None:
        return
    try:
        import torch
    except Exception:
        return
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _infer(policy, model_obs: dict):
    """One forward with autograd disabled."""
    try:
        import torch
    except Exception:
        return policy.get_action(model_obs)
    with torch.inference_mode():
        return policy.get_action(model_obs)


def _release_cached_gpu_memory() -> None:
    """Return the caching allocator's unused blocks. Live weights are kept."""
    import gc

    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _install_rot6d_guard() -> None:
    """Replace GR00T's 6D decoder so a collapsed rotation does not call a failing SVD.

    ``Rotation.from_matrix`` runs ``numpy.linalg.svd`` when the 3x3 is not already
    orthogonal. A zero or parallel 6D row divides by zero inside Gram-Schmidt, the
    matrix becomes NaN, and that SVD raises ``SVD did not converge``.
    """
    from gr00t.data.state_action.pose import EndEffectorPose

    EndEffectorPose._rot6d_to_matrix = staticmethod(_rot6d_to_matrix)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """First two rows of a rotation matrix, completed to a right-handed basis."""
    raw = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row1 = _unit_row(raw[0], np.array([1.0, 0.0, 0.0]))
    row2 = raw[1] - np.dot(row1, raw[1]) * row1
    if not np.isfinite(row2).all() or float(np.linalg.norm(row2)) < 1e-8:
        helper = np.array([1.0, 0.0, 0.0]) if abs(float(row1[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        row2 = helper - np.dot(row1, helper) * row1
    row2 = row2 / np.linalg.norm(row2)
    return np.vstack((row1, row2, np.cross(row1, row2)))


def _unit_row(vector: np.ndarray, default: np.ndarray) -> np.ndarray:
    if not np.isfinite(vector).all():
        return default.copy()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return default.copy()
    return vector / norm


def _language(instructions: list[str]) -> dict[str, list[list[str]]]:
    """Batch instructions as one timestep under the DROID language key."""
    return {"annotation.language.language_instruction": [[text] for text in instructions]}


def _time(values: np.ndarray) -> np.ndarray:
    """Insert the single GR00T timestep axis. States arrive as [N, D]."""
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim == 2:
        array = array[:, None, :]
    return array


def _video(frames: np.ndarray) -> np.ndarray:
    """Insert the timestep axis and keep RGB frames as uint8 [N, 1, H, W, C]."""
    array = np.asarray(frames)
    if array.ndim == 3:
        array = array[None]
    if array.ndim == 4:
        array = array[:, None]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _eef_9d(pose: np.ndarray) -> np.ndarray:
    """Position [m] plus the first two rotation-matrix rows from a wxyz quaternion.

    GR00T's ``xyz+rot6d`` layout is ``rotation_matrix[:2, :].flatten()``, not the
    first two columns.
    """
    position = np.nan_to_num(np.asarray(pose[..., :3], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    quat = np.asarray(pose[..., 3:7], dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    valid = np.isfinite(quat).all(axis=-1, keepdims=True) & (norm > 1e-8)
    quat = np.where(valid, quat / np.maximum(norm, 1e-8), np.array([1.0, 0.0, 0.0, 0.0]))
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    row0 = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), axis=-1)
    row1 = np.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), axis=-1)
    return np.concatenate((position, row0, row1), axis=-1).astype(np.float32)
