#!/usr/bin/env bash
set -euo pipefail

cd /root/machine-learning-for-finance/Final/scripts/pipeline_split

MASTER_LOG="/root/machine-learning-for-finance/Final/outputs/pipeline_runs/logs/master_stages.log"
mkdir -p "$(dirname "${MASTER_LOG}")"

echo "[MASTER] start $(date '+%F %T')" | tee -a "${MASTER_LOG}"

run_stage() {
  local stage_script="$1"
  echo "[MASTER] running ${stage_script} at $(date '+%F %T')" | tee -a "${MASTER_LOG}"
  if python "${stage_script}" >> "${MASTER_LOG}" 2>&1; then
    echo "[MASTER] ${stage_script} success $(date '+%F %T')" | tee -a "${MASTER_LOG}"
  else
    echo "[MASTER] ${stage_script} failed $(date '+%F %T')" | tee -a "${MASTER_LOG}"
  fi
}

# 为稳定性按顺序执行，避免并行争抢内存/内核导致崩溃
run_stage run_fig6_stage.py
run_stage run_fig7_stage.py
run_stage run_fig8_stage.py
run_stage run_table1_stage.py

echo "[MASTER] all done $(date '+%F %T')" | tee -a "${MASTER_LOG}"
