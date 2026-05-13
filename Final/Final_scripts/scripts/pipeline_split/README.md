# Split Pipeline (GPU-first)

This folder splits execution into independent `.py` entrypoints.

## Stages

- `run_core_stage.py` - core pipeline, heavy figures/tables disabled
- `run_fig6_stage.py` - Figure 6 only
- `run_fig7_stage.py` - Figure 7 only
- `run_fig8_stage.py` - Figure 8 only
- `run_table1_stage.py` - Table I only

## How outputs are organized

All stage outputs are categorized under:

- `Final/outputs/pipeline_runs/logs/` - stage logs
- `Final/outputs/pipeline_runs/notebooks/` - stage notebook snapshots
- `Final/outputs/pipeline_runs/meta/` - stage metadata/status files

## Notes about GPU

- Model inference/training uses CUDA (`fp16` enabled by default).
- Tabular `pandas` `groupby/rolling/merge` parts are still CPU-bound in current notebook logic.
- For full GPU tabular acceleration (cuDF), results can have small numeric differences and require validation.
