# Fine-tuning

Evaluation does not run this directory.

Run `scripts/fetch_weights.sh` to put the released `nvidia/GR00T-N1.7-DROID` checkpoint, including the backbone, in `weights/`. `agent.yaml` lists each file's sha256. The embodiment tag is `OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`.

To fine-tune further, start from that checkpoint with Isaac-GR00T's N1.7 trainer and the dataset from `task-franka-tabletop/scripts/record_demos.py`. Replace `weights/` with the new checkpoint and run `udcli submission hash --path .` to update the hashes in `agent.yaml`.
