#!/usr/bin/env bash
set -u

cd /data1/nHome1/xieqihu/homework/project || exit 1

OUT_DIR="outputs/dcgan_bce_ema_ch128_100"
LOG="${OUT_DIR}/train.log"
THRESHOLD_MIB=20000

mkdir -p "${OUT_DIR}"

{
  echo "[$(date)] queued ch=128 experiment from script"
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    echo "[$(date)] gpu memory used: ${USED} MiB"
    if [ "${USED}" -lt "${THRESHOLD_MIB}" ]; then
      break
    fi
    sleep 300
  done

  echo "[$(date)] starting training"
  conda run -n nern python train.py \
    --gan-type dcgan \
    --loss bce \
    --ema \
    --out-dir "${OUT_DIR}" \
    --epochs 100 \
    --batch-size 128 \
    --g-channels 128 \
    --d-channels 128 \
    --device cuda \
    --num-workers 2 \
    --checkpoint-every 10
  STATUS=$?
  echo "[$(date)] training exit status: ${STATUS}"
} >> "${LOG}" 2>&1
