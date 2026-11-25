# unidroid-template-groot17

Submission template for the `groot17` brain on the Franka tabletop task. It runs the released `nvidia/GR00T-N1.7-DROID` checkpoint. Copy your trained weights into `weights/` and submit.

## Quickstart

1. Put the weights in `weights/`. For the released checkpoint, run `scripts/fetch_weights.sh`. For your own fine-tune, replace the contents of `weights/` with your checkpoint (see `train/README.md`).
2. Update the hashes in `agent.yaml`:

   ```bash
   udcli submission hash --path .
   ```

3. Check the folder and run the brain locally:

   ```bash
   brain-comm check --submission .
   mkdir -p /tmp/brain
   brain-comm serve --submission . --socket-dir /tmp/brain --replicas 1
   ```

4. Submit. See the Miner quick start in the `unidroid-cli` README (`udcli submission check`, then `udcli submission submit`). Do not change the pushed weights after submitting. A copy of the unmodified released weights is rejected, but a fine-tune is accepted.

## What you may modify

Everything except the file layout that `udcli submission check` enforces.

| Path | Purpose |
| --- | --- |
| `brain/policy.py` | Inference: loads the checkpoint and answers each observation. |
| `brain/adapt.py` | Observation and action conversion. |
| `train/` | Fine-tuning notes or scripts. Uploaded in the code bundle, not evaluated. |
| `tests/` | Unit tests. Run with `pytest`. |
| `agent.yaml` | Brain config. The `weights:` block is rewritten by `udcli submission hash`. |
| `weights/` | Checkpoint. Ignored by Git and pushed to your Hugging Face repo at submit time. |

## Brain specifics

`agent.yaml` sets `embodiment_tag` to `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`.

The policy reads the tabletop cameras, joints, gripper, and end-effector pose, and turns the model's relative joint delta plus absolute gripper into this task's 8-D joint command. Loading the checkpoint requires the GR00T N1.7 runtime from the brain image.

The released checkpoint is at `https://huggingface.co/nvidia/GR00T-N1.7-DROID`. Do not leave small Git placeholder files in `weights/`; `scripts/fetch_weights.sh` replaces them.
