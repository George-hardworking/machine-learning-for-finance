#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/bin/activate 5020_env
cd /root/machine-learning-for-finance/Final/scripts

NOTEBOOK_IN="final_improve.ipynb"
NOTEBOOK_OUT="final_improve_output.ipynb"
LOG_FILE="run_live.log"
PID_FILE="run_live.pid"

# 启动前清理旧标记，避免误判
rm -f "${PID_FILE}" "${NOTEBOOK_OUT}"
: > "${LOG_FILE}"

# 修复历史输出里可能残留的 nbformat 非法字段，避免 papermill 启动即失败
python - <<'PY'
import json
from pathlib import Path

p = Path("final_improve.ipynb")
nb = json.loads(p.read_text())
dirty = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    outputs = cell.get("outputs", [])
    for out in outputs:
        ot = out.get("output_type")
        if ot == "stream" and "name" not in out:
            out["name"] = "stdout"
            dirty = True
        if ot == "execute_result" and "metadata" not in out:
            out["metadata"] = {}
            dirty = True
if dirty:
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
PY

export CPU_THREADS="${CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCH_COMPILE="${TORCH_COMPILE:-0}"
export PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-1024}"
export USE_FP16_INFER="${USE_FP16_INFER:-1}"
export FIG6_ENABLE="${FIG6_ENABLE:-0}"
export FIG7_ENABLE="${FIG7_ENABLE:-0}"
export FIG8_ENABLE="${FIG8_ENABLE:-0}"
export TABLE1_ENABLE="${TABLE1_ENABLE:-0}"
export FIG_OUT_DIR="${FIG_OUT_DIR:-/root/autodl-tmp/outputs/pipeline_runs/figures}"
export TABLE_OUT_DIR="${TABLE_OUT_DIR:-/root/autodl-tmp/outputs/pipeline_runs/tables}"
export FIG_DATA_OUT_DIR="${FIG_DATA_OUT_DIR:-$TABLE_OUT_DIR}"
export FIG7_MAX_LAG="${FIG7_MAX_LAG:-6}"
export FIG7_MAX_TRAIN_ROWS="${FIG7_MAX_TRAIN_ROWS:-400000}"
export FIG7_PRED_CHUNK="${FIG7_PRED_CHUNK:-200000}"

KERNEL_NAME="${KERNEL_NAME:-5020_env}"

python - <<'PY'
import json
import subprocess
import sys
kernel = subprocess.check_output(["jupyter", "kernelspec", "list", "--json"], text=True)
names = set(json.loads(kernel)["kernelspecs"].keys())
target = __import__("os").environ.get("KERNEL_NAME", "5020_env")
if target not in names:
    print(f"[ERROR] kernel '{target}' not found. Available: {sorted(names)}")
    sys.exit(1)
print(f"[INFO] using kernel: {target}")
PY

echo $$ > "${PID_FILE}"
echo "[INFO] start at $(date '+%F %T')" | tee -a "${LOG_FILE}"
echo "[INFO] TORCH_COMPILE=${TORCH_COMPILE} | DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS} | CPU_THREADS=${CPU_THREADS} | PRED_BATCH_SIZE=${PRED_BATCH_SIZE} | FP16=${USE_FP16_INFER}" | tee -a "${LOG_FILE}"

run_once() {
  stdbuf -oL -eL papermill "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" \
    -k "${KERNEL_NAME}" \
    --log-output \
    --progress-bar \
    2>&1 | tee -a "${LOG_FILE}"
}

set +e
run_once
rc=$?
set -e

if [ $rc -ne 0 ] && [ "${AUTO_FALLBACK_ON_FAIL:-1}" = "1" ]; then
  echo "[WARN] fast profile failed (exit=$rc), auto fallback to stable profile..." | tee -a "${LOG_FILE}"
  export CPU_THREADS=1
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export DATALOADER_NUM_WORKERS=0
  export PRED_BATCH_SIZE=512
  export USE_FP16_INFER=0
  export FIG6_ENABLE=0
  export FIG7_ENABLE=0
  export FIG8_ENABLE=0
  export TABLE1_ENABLE=0
  echo "[INFO] fallback profile: workers=${DATALOADER_NUM_WORKERS}, threads=${CPU_THREADS}, batch=${PRED_BATCH_SIZE}, fp16=${USE_FP16_INFER}" | tee -a "${LOG_FILE}"
  run_once
  rc=$?
fi

exit $rc
