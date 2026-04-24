#!/usr/bin/env bash
# GCP bootstrap for Ministral 3 3B LoRA training.
#
# Usage (from the engram repo root, after quota is approved):
#   bash experiments/phase2_training/gcp_bootstrap.sh
#
# What it does:
# 1. Creates an A2 ultragpu-1g spot VM with Deep Learning VM image.
# 2. Copies training data + script up.
# 3. Installs torch + transformers + peft + trl if missing.
# 4. Runs training.
# 5. Copies adapter back to engram/experiments/phase2_training/adapter/
# 6. Destroys the VM.
#
# Idempotent-ish: re-running picks up where it left off via the
# VM name (if VM exists, skips creation).

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-merken-training-28424}"
ZONE="${ZONE:-us-central1-a}"
INSTANCE="${INSTANCE:-merken-phase2-a100}"
MACHINE_TYPE="a2-ultragpu-1g"
IMAGE_FAMILY="common-cu124-debian-11-py310"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE_GB=100
DATA_LOCAL="${DATA_LOCAL:-data/phase2_sft/train.jsonl}"
DATA_REMOTE="/tmp/train.jsonl"
SCRIPT_LOCAL="experiments/phase2_training/train_ministral_lora.py"
SCRIPT_REMOTE="/tmp/train_ministral_lora.py"
ADAPTER_REMOTE="/tmp/output/adapter_final"
ADAPTER_LOCAL="experiments/phase2_training/adapter"

echo "[gcp] project=$PROJECT_ID zone=$ZONE instance=$INSTANCE"
echo "[gcp] data: $DATA_LOCAL -> $DATA_REMOTE"
echo "[gcp] script: $SCRIPT_LOCAL -> $SCRIPT_REMOTE"

if [[ ! -f "$DATA_LOCAL" ]]; then
  echo "[gcp] ERROR: training data not found at $DATA_LOCAL" >&2
  echo "[gcp] Run build_training_data.py first." >&2
  exit 1
fi

# --- Step 1: create VM if missing --------------------------------------------
if gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
  echo "[gcp] VM $INSTANCE already exists; reusing."
else
  echo "[gcp] Creating VM $INSTANCE (spot, A100 80GB)..."
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=nvidia-a100-80gb,count=1" \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform
  echo "[gcp] VM created. Waiting 30s for Nvidia driver install..."
  sleep 30
fi

# Wait until SSH is ready.
echo "[gcp] Waiting for SSH to be ready..."
for i in 1 2 3 4 5 6; do
  if gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --command="echo ok" --quiet &>/dev/null; then
    break
  fi
  echo "[gcp] SSH not ready, waiting 15s (attempt $i/6)..."
  sleep 15
done

# --- Step 2: upload data + script --------------------------------------------
echo "[gcp] Uploading training data + script..."
gcloud compute scp "$DATA_LOCAL" "$INSTANCE:$DATA_REMOTE" --zone="$ZONE" --project="$PROJECT_ID"
gcloud compute scp "$SCRIPT_LOCAL" "$INSTANCE:$SCRIPT_REMOTE" --zone="$ZONE" --project="$PROJECT_ID"

# --- Step 3: install deps on VM ---------------------------------------------
echo "[gcp] Installing deps on VM..."
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --command="
  set -e
  nvidia-smi | head -15
  pip install --quiet --upgrade pip
  pip install --quiet 'torch' 'transformers>=4.45' 'peft>=0.14' 'trl>=0.12' \
    'datasets' 'accelerate' 'bitsandbytes' 'sentencepiece'
  pip list | grep -iE 'torch|transformers|peft|trl|datasets' | head
"

# --- Step 4: run training ---------------------------------------------------
echo "[gcp] Starting LoRA training on VM..."
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --command="
  set -e
  cd /tmp
  python3 train_ministral_lora.py \
    --data $DATA_REMOTE \
    --output-dir /tmp/output 2>&1 | tee /tmp/train.log
"

# --- Step 5: copy adapter back ----------------------------------------------
echo "[gcp] Copying adapter back to $ADAPTER_LOCAL..."
mkdir -p "$ADAPTER_LOCAL"
gcloud compute scp --recurse "$INSTANCE:$ADAPTER_REMOTE" "$ADAPTER_LOCAL" \
  --zone="$ZONE" --project="$PROJECT_ID"
gcloud compute scp "$INSTANCE:/tmp/train.log" "$ADAPTER_LOCAL/train.log" \
  --zone="$ZONE" --project="$PROJECT_ID"

# --- Step 6: destroy VM -----------------------------------------------------
echo "[gcp] Deleting VM $INSTANCE to stop billing..."
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" --quiet

echo "[gcp] DONE. Adapter at: $ADAPTER_LOCAL"
