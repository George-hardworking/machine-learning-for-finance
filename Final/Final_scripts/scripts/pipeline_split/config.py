from pathlib import Path

BASE_DIR = Path("/root/machine-learning-for-finance/Final/scripts")
NOTEBOOK_IN = BASE_DIR / "final_improve.ipynb"
RUNNER = BASE_DIR / "run_papermill.sh"

OUTPUT_ROOT = Path("/root/machine-learning-for-finance/Final/outputs/pipeline_runs")
LOG_DIR = OUTPUT_ROOT / "logs"
NB_DIR = OUTPUT_ROOT / "notebooks"
META_DIR = OUTPUT_ROOT / "meta"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"

for d in (OUTPUT_ROOT, LOG_DIR, NB_DIR, META_DIR, FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Fast GPU-friendly profile with automatic fallback handled by run_papermill.sh
FAST_PROFILE = {
    "CPU_THREADS": "2",
    "DATALOADER_NUM_WORKERS": "2",
    "PRED_BATCH_SIZE": "1024",
    "USE_FP16_INFER": "1",
    "TORCH_COMPILE": "0",
}

