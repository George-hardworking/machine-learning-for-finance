# Project Handover (Final)

## 1) Project Objective

This project reproduces and extends the paper-style CNN return prediction/backtest pipeline on CRSP-like daily stock data.

Main goals completed in this phase:
- Make notebook execution more stable on rented GPU servers.
- Support cached re-runs (avoid recomputing expensive steps).
- Split heavy stages so figures/tables can be run independently.
- Save outputs with clear horizon suffixes (`I5`, `I20`, `I60`) to avoid overwrite.
- Export structured CSV data for downstream report-writing agents.

---

## 2) Current Runtime Status

At handover time:
- `fig7` stage has completed successfully (simple Figure 7 outputs + CSVs are present).
- `fig6` stage via papermill is unstable on this server profile (`DeadKernelError` in Cell 16).
- Figure 6 CSVs were successfully generated via a cache-only lightweight exporter script.
- Figure 6 CSV status: complete (`9/9` files across `I5/I20/I60 x cnn/mom/ma`).

Check live status with:
- `pgrep -af "run_fig6_stage.py|run_papermill.sh|papermill .*final_improve.ipynb"`
- Read latest logs in `Final/outputs/pipeline_runs/logs`.

---

## 3) Key Directories (What they mean)

- `Final/scripts`
  - Core notebook and all runner/orchestration scripts.
- `Final/outputs` (symlink to `/root/autodl-tmp/outputs`)
  - All generated artifacts (cache, models, figures, tables, stage logs).
- `/root/autodl-tmp/outputs/cache`
  - Heavy intermediate cache (largest files, e.g. cleaned panel/cache arrays).
- `/root/autodl-tmp/outputs/models`
  - Trained checkpoints and normalization stats.
- `/root/autodl-tmp/outputs/pipeline_runs`
  - Stage-style artifacts (per-run logs/meta/notebook snapshots/figures/tables).
- `/root/autodl-tmp/outputs/ppt_images`
  - Presentation-ready PPT images and small CSV metrics.

---

## 4) File-by-File Guide (Scripts)

### Core notebook
- `Final/scripts/final_improve.ipynb`
  - Main end-to-end pipeline notebook.
  - Used by all stage runners via `papermill`.
  - Contains env-gated sections (`FIG6_ENABLE`, `FIG7_ENABLE`, `FIG8_ENABLE`, `TABLE1_ENABLE`).

### Main execution wrapper
- `Final/scripts/run_papermill.sh`
  - Primary execution entry used in this phase.
  - Responsibilities:
    - Activate kernel env (`5020_env`).
    - Patch notebook JSON compatibility issues before run.
    - Set runtime env (threads/workers/batch/fp16/stage switches).
    - Run papermill with a fast profile; fallback to safer profile if needed.
  - Use when: launching full or stage-filtered notebook execution.

### Split-stage runner system
- `Final/scripts/pipeline_split/config.py`
  - Central paths and fast profile defaults.
- `Final/scripts/pipeline_split/run_stage.py`
  - Generic single-stage runner.
  - Enables one heavy stage at a time.
  - Writes stage `log/meta/snapshot notebook`.
  - Has cache-hit skip behavior for expected outputs.
- `Final/scripts/pipeline_split/run_fig6_stage.py`
  - Runs Figure 6 stage only.
- `Final/scripts/pipeline_split/run_fig7_stage.py`
  - Runs Figure 7 stage only.
- `Final/scripts/pipeline_split/run_fig8_stage.py`
  - Runs Figure 8 stage only.
- `Final/scripts/pipeline_split/run_table1_stage.py`
  - Runs Table 1 stage only.
- `Final/scripts/pipeline_split/run_core_stage.py`
  - Runs "core only" mode (heavy figure/table sections disabled).
- `Final/scripts/pipeline_split/run_all_stages.sh`
  - Sequential stage launcher.
- `Final/scripts/pipeline_split/export_fig6_csv_from_cache.py`
  - Lightweight fallback exporter for Figure 6 CSVs.
  - Reads cache directly and bypasses heavy notebook execution that was crashing.

### Reporting / alternate flow helpers
- `Final/scripts/final_report.ipynb`
  - Lightweight report notebook intended to read existing outputs.
- `Final/scripts/run_core_pipeline.py`
  - Alternative orchestration path for core-first execution strategy.
- `Final/scripts/run_core.sh`
  - Shell entry for core pipeline orchestration.
- `Final/scripts/status_core.sh`
  - Quick status/log helper for core pipeline run.

### Utility files
- `Final/scripts/fix_notebook.py`
  - Notebook format/compatibility helper.
- `Final/scripts/run_live.log`, `Final/scripts/run_live.pid`
  - Current/last run status markers for `run_papermill.sh`.

### Legacy/secondary files (can usually ignore)
- `Final/scripts/final.ipynb`
- `Final/scripts/final_core.ipynb`
- `Final/scripts/final_core_output.ipynb`
- `Final/scripts/final_improve_documented.ipynb`
- `Final/scripts/final_improve_documented.py`
- `Final/scripts/_patch_smoke_test.py`
- `Final/scripts/core_guardian.sh`

---

## 5) Outputs Produced and Their Meaning

## Figure-related outputs
- `Final/outputs/pipeline_runs/figures/fig6_I5.png`, `fig6_I20.png`, `fig6_I60.png`
  - Figure 6 images by horizon.
- `Final/outputs/pipeline_runs/figures/fig7_simple_I5.png`, `fig7_simple_I20.png`, `fig7_simple_I60.png`
  - Figure 7 simple version images by horizon.
- `fig7_fixed_I20.png` (if generated in a dedicated fixed run)
  - Figure 7 fixed version image (CNN + linear + baselines for I20).

## CSV outputs for AI-friendly reporting
- Figure 7 simple decile CSVs:
  - `fig7_simple_I5_{cnn,mom,ma,reversal}.csv`
  - `fig7_simple_I20_{cnn,mom,ma,reversal}.csv`
  - `fig7_simple_I60_{cnn,mom,ma,reversal}.csv`
- Figure 6 decile CSVs (now confirmed generated):
  - `fig6_I5_{cnn,mom,ma}.csv`
  - `fig6_I20_{cnn,mom,ma}.csv`
  - `fig6_I60_{cnn,mom,ma}.csv`
- Table 1 CSVs:
  - `table1_I5.csv`, `table1_I20.csv`, `table1_I60.csv`

## PPT assets
- `Final/outputs/ppt_images/ppt15_cumulative_returns_L5.png` etc.
  - Net (after transaction cost) cumulative return images.
- `Final/outputs/ppt_images/ppt15_cumulative_returns_gross_L5.png` etc.
  - Gross (before transaction cost) cumulative return images.
- `ppt16_classification_metrics.csv`
  - Classification summary metrics for PPT usage.

---

## 6) Logs and Debug Traces

- Stage logs: `Final/outputs/pipeline_runs/logs/*.log`
- Stage metadata: `Final/outputs/pipeline_runs/meta/*.txt`
- Stage notebook snapshots: `Final/outputs/pipeline_runs/notebooks/*.ipynb`

Recommended debug workflow:
1. Open latest `meta` file to see `return_code` and corresponding log path.
2. Inspect matching `logs/*.log` for crash location (cell index / error type).
3. Re-run only the failed stage with split runner.

---

## 7) Typical Operations (When to use what)

### Run only Figure 7
Use when you need Figure 7 images/CSVs quickly without rerunning other heavy sections.

Command:
`python Final/scripts/pipeline_split/run_fig7_stage.py`

### Run only Figure 6
Use when Figure 6 outputs are missing or need refresh.

Command:
`python Final/scripts/pipeline_split/run_fig6_stage.py`

### Run all staged post-analysis
Use when you want full post-processing output refresh.

Command:
`bash Final/scripts/pipeline_split/run_all_stages.sh`

### Check if a run is alive
Use during long runs to avoid blind waiting.

Command:
`pgrep -af "run_papermill.sh|papermill .*final_improve.ipynb"`

---

## 8) Data Transfer / Export Notes

Already validated working transfer commands:
- `Final/scripts` (core code)
- `/root/autodl-tmp/outputs` (results + cache + models + logs)

This is sufficient for local takeover in most cases.

`full_backup_no_raw_*.tar.part-*` is optional:
- Use only if you want a single archive snapshot for cold backup.
- Otherwise it is largely redundant after direct `rsync` of scripts + outputs.

---

## 9) Known Risks / Open Items

- Figure 6 stage can still hit intermittent kernel death on this server profile.
  - Mitigation: use `export_fig6_csv_from_cache.py` for CSV outputs; rely on existing PNG outputs.
- Figure 7 fixed (`fig7_fixed_I20`) may need explicit targeted run if not present.
- Ensure latest local copy includes any outputs generated after the last `rsync`.

---

## 10) Minimal Local Continuation Checklist

1. Confirm local copy contains:
   - `Final/scripts/final_improve.ipynb`
   - `outputs/cache/*`
   - `outputs/models/*`
   - `outputs/pipeline_runs/*`
2. In local environment, point code to copied `outputs` path.
3. Start with stage-level execution (not full run) for stability:
   - first `fig6` or `fig7`, then other stages as needed.
4. For report agent ingestion, prioritize `outputs/pipeline_runs/tables/*.csv`.

