# Local Agent SOP (Quick Takeover)

## Mission
Continue this project locally with minimum risk, using existing cached artifacts and stage-level execution.

---

## Scope To Use
- Code: `Final/scripts`
- Artifacts: `outputs` (copied from `/root/autodl-tmp/outputs`)
- Primary notebook: `Final/scripts/final_improve.ipynb`

Do **not** rely on `raw` unless explicitly needed.

---

## First 60-Second Checks
1. Confirm these folders exist locally:
   - `Final/scripts`
   - `outputs/cache`
   - `outputs/models`
   - `outputs/pipeline_runs`
2. Confirm key cache files exist:
   - `outputs/cache/cleaned_data_all.pkl`
   - `outputs/cache/preds_I5.npy`, `preds_I20.npy`, `preds_I60.npy`
3. Confirm model checkpoints exist in `outputs/models`.

If any are missing, stop and restore from backup before running.

---

## Execution Policy
- Prefer stage-level runs, not full notebook reruns.
- Run only what is missing.
- Reuse cached outputs whenever possible.

Primary stage commands:
- Figure 6: `python Final/scripts/pipeline_split/run_fig6_stage.py`
- Figure 7: `python Final/scripts/pipeline_split/run_fig7_stage.py`
- Figure 8: `python Final/scripts/pipeline_split/run_fig8_stage.py`
- Table 1: `python Final/scripts/pipeline_split/run_table1_stage.py`

Fallback command (Figure 6 CSV only, cache-based):
- `python Final/scripts/pipeline_split/export_fig6_csv_from_cache.py`

---

## Output Targets (must exist)

### Figure 6
- `outputs/pipeline_runs/figures/fig6_I5.png`
- `outputs/pipeline_runs/figures/fig6_I20.png`
- `outputs/pipeline_runs/figures/fig6_I60.png`
- `outputs/pipeline_runs/tables/fig6_I5_{cnn,mom,ma}.csv`
- `outputs/pipeline_runs/tables/fig6_I20_{cnn,mom,ma}.csv`
- `outputs/pipeline_runs/tables/fig6_I60_{cnn,mom,ma}.csv`
Status in this handover: CSV set generated successfully (9/9).

### Figure 7 (simple)
- `outputs/pipeline_runs/figures/fig7_simple_I5.png`
- `outputs/pipeline_runs/figures/fig7_simple_I20.png`
- `outputs/pipeline_runs/figures/fig7_simple_I60.png`
- `outputs/pipeline_runs/tables/fig7_simple_I5_{cnn,mom,ma,reversal}.csv`
- `outputs/pipeline_runs/tables/fig7_simple_I20_{cnn,mom,ma,reversal}.csv`
- `outputs/pipeline_runs/tables/fig7_simple_I60_{cnn,mom,ma,reversal}.csv`

### Table 1
- `outputs/pipeline_runs/tables/table1_I5.csv`
- `outputs/pipeline_runs/tables/table1_I20.csv`
- `outputs/pipeline_runs/tables/table1_I60.csv`

### PPT assets
- `outputs/ppt_images/ppt15_cumulative_returns_L5.png` (net, after costs)
- `outputs/ppt_images/ppt15_cumulative_returns_gross_L5.png` (gross, before costs)
- similarly for `L20`, `L60`

---

## Debug SOP
If a stage fails:
1. Read latest meta file in `outputs/pipeline_runs/meta`.
2. Open referenced log in `outputs/pipeline_runs/logs`.
3. Identify failing cell and error type.
4. Re-run only that stage.

Known common failure:
- `DeadKernelError` during heavy notebook cells.
  - Mitigation: rerun isolated stage; if Figure 6 still fails, use `export_fig6_csv_from_cache.py`.

---

## Reporting SOP (for downstream AI)
- Prefer CSVs under `outputs/pipeline_runs/tables`.
- Use PNGs only for visual confirmation.
- For narrative consistency:
  - `ppt15_cumulative_returns_*.png` = net after trading costs
  - `ppt15_cumulative_returns_gross_*.png` = gross before trading costs

---

## Final Verification Checklist
- [ ] All required PNG targets exist.
- [ ] All required CSV targets exist.
- [ ] Latest stage meta files show `return_code=0`.
- [ ] Logs are archived in `outputs/pipeline_runs/logs`.

