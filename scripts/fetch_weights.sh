#!/usr/bin/env bash
# Download the released nvidia/GR00T-N1.7-DROID checkpoint into weights/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
hf download nvidia/GR00T-N1.7-DROID --local-dir weights
# The Hugging Face cache folder is not part of the checkpoint and fails the weight hash check.
rm -rf weights/.cache
