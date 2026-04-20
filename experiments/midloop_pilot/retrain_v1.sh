#!/usr/bin/env bash
# End-to-end retrain for midloop v1.
#
# Preconditions:
#   - scaleup_out_hf/midloop_v1_hf.jsonl exists (output of scaleup_phase1_hf.py)
#   - training_data/midloop_v0.jsonl exists (original v0)
#
# Runs:
#   1. Merge v0 + v1_hf -> training_data/midloop_v1.jsonl
#   2. Split (10% protocol-level holdout) -> midloop_v1_{train,test}.jsonl
#   3. nanoGPT prepare.py on v1 split
#   4. train_midloop.py with config/train_midloop_v1.py
#   5. train-vs-val F1 diagnostic on the new ckpt

set -euo pipefail
cd "$(dirname "$0")/../.."  # engram repo root

NANO=${NANOGPT_REPO:-$HOME/Desktop/Personal/Projects/nanoGPT}

echo "=== step 1/5: merge v0 + v1_hf ==="
python -m experiments.midloop_pilot.merge_datasets

echo "=== step 2/5: split v1 ==="
# Count protocols in the merged set to pick a reasonable test_protocols value
N_PROTO=$(python -c "
import json
pids = set()
with open('experiments/midloop_pilot/training_data/midloop_v1.jsonl') as f:
    for l in f:
        pids.add(json.loads(l)['metadata']['protocol_id'])
print(len(pids))
")
# Guardrail: tiny datasets produce meaningless splits. 20 is the smallest
# we'd try to train on (would yield 2 test protocols at 10%).
if [ "$N_PROTO" -lt 20 ]; then
    echo "ERROR: N_PROTO=$N_PROTO too small for a meaningful split (need >= 20)" >&2
    exit 1
fi
N_TEST=$(( (N_PROTO * 10 + 99) / 100 ))  # ceil(10%)
echo "  total protocols: $N_PROTO, test-protocols: $N_TEST"
python -m experiments.midloop_pilot.split_train_test \
  --in experiments/midloop_pilot/training_data/midloop_v1.jsonl \
  --train experiments/midloop_pilot/training_data/midloop_v1_train.jsonl \
  --test experiments/midloop_pilot/training_data/midloop_v1_test.jsonl \
  --test-protocols "$N_TEST"

echo "=== step 3/5: nanoGPT prepare v1 ==="
(cd "$NANO" && python data/midloop_v1/prepare.py)

echo "=== step 4/5: train v1 ==="
(cd "$NANO" && python train_midloop.py config/train_midloop_v1.py)

echo "=== step 5/5: train-vs-val diagnostic ==="
(cd "$NANO" && python eval_midloop_train.py out-midloop-v1/ckpt.pt)

echo "=== done ==="
