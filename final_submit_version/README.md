# Final Reproduction

This folder is organized as a standalone, submission-ready reproduction project.

## Layout

- `notebooks/final_improve.ipynb`: main notebook (closest to paper logic).
- `notebooks/archive/`: legacy notebook versions kept for history.
- `scripts/`: runnable orchestration scripts.
- `scripts/pipeline_split/`: stage-based runners (`fig6`, `fig7`, `fig8`, `table1`, `core`).
- `outputs/`: local artifacts (cache/models/figures/tables/logs).

## Quick Start

From this folder:

- Run one stage:
  - `python scripts/pipeline_split/run_fig6_stage.py`
  - `python scripts/pipeline_split/run_fig7_stage.py`
  - `python scripts/pipeline_split/run_fig8_stage.py`
  - `python scripts/pipeline_split/run_table1_stage.py`
- Run all post-analysis stages:
  - `bash scripts/pipeline_split/run_all_stages.sh`

## Notes

- Paths were rewritten to be project-relative for local and cloud portability.
- `outputs/` is intended for local storage and is usually excluded from commits.
