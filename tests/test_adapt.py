# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "brain" / "adapt.py"
_SPEC = importlib.util.spec_from_file_location("groot17_adapt", _PATH)
adapt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapt)


def test_resize_keeps_batch_shape():
    images = np.zeros((2, 720, 1280, 3), dtype=np.uint8)
    assert adapt.resize_hwc(images).shape == (2, *adapt.IMAGE_HW, 3)
    assert adapt.resize_hwc(images[0]).shape == (*adapt.IMAGE_HW, 3)


def test_joint_action_chunk_joins_joints_and_gripper():
    action = {
        "action.joint_position": np.zeros((3, 16, 7), dtype=np.float32),
        "action.gripper_position": np.ones((3, 16), dtype=np.float32),
    }
    chunk = adapt.joint_action_chunk((action, {}), 8)
    assert chunk.shape == (3, 8, 8)
    assert (chunk[..., 7] == 1).all()
