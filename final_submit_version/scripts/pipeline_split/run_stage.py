#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, FAST_PROFILE, FIG_DIR, LOG_DIR, META_DIR, NB_DIR, NOTEBOOK_IN, RUNNER, TABLE_DIR


def expected_outputs(stage: str):
    fig_dir = FIG_DIR
    tab_dir = TABLE_DIR
    if stage == "fig6":
        return [
            fig_dir / "fig6_I5.png",
            fig_dir / "fig6_I20.png",
            fig_dir / "fig6_I60.png",
            tab_dir / "fig6_I5_cnn.csv",
            tab_dir / "fig6_I5_mom.csv",
            tab_dir / "fig6_I5_ma.csv",
            tab_dir / "fig6_I20_cnn.csv",
            tab_dir / "fig6_I20_mom.csv",
            tab_dir / "fig6_I20_ma.csv",
            tab_dir / "fig6_I60_cnn.csv",
            tab_dir / "fig6_I60_mom.csv",
            tab_dir / "fig6_I60_ma.csv",
        ]
    if stage == "fig7":
        return [
            fig_dir / "fig7_fixed_I20.png",
            fig_dir / "fig7_simple_I5.png",
            fig_dir / "fig7_simple_I20.png",
            fig_dir / "fig7_simple_I60.png",
            tab_dir / "fig7_fixed_I20_cnn.csv",
            tab_dir / "fig7_fixed_I20_linear.csv",
            tab_dir / "fig7_fixed_I20_reversal.csv",
            tab_dir / "fig7_fixed_I20_mom.csv",
            tab_dir / "fig7_fixed_I20_ma.csv",
            tab_dir / "fig7_simple_I5_cnn.csv",
            tab_dir / "fig7_simple_I5_reversal.csv",
            tab_dir / "fig7_simple_I5_mom.csv",
            tab_dir / "fig7_simple_I5_ma.csv",
            tab_dir / "fig7_simple_I20_cnn.csv",
            tab_dir / "fig7_simple_I20_reversal.csv",
            tab_dir / "fig7_simple_I20_mom.csv",
            tab_dir / "fig7_simple_I20_ma.csv",
            tab_dir / "fig7_simple_I60_cnn.csv",
            tab_dir / "fig7_simple_I60_reversal.csv",
            tab_dir / "fig7_simple_I60_mom.csv",
            tab_dir / "fig7_simple_I60_ma.csv",
        ]
    if stage == "fig8":
        return [fig_dir / "fig8_I20.png"]
    if stage == "table1":
        return [tab_dir / "table1_I20.csv", tab_dir / "table1_I5.csv", tab_dir / "table1_I60.csv"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["core", "fig6", "fig7", "fig8", "table1"])
    args = parser.parse_args()

    stage = args.stage
    force = os.environ.get("FORCE_RERUN", "0") == "1"

    outs = expected_outputs(stage)
    if (not force) and outs and all(p.exists() for p in outs):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_meta = META_DIR / f"{stage}_{ts}.txt"
        stage_meta.write_text(
            "\n".join(
                [
                    f"stage={stage}",
                    "return_code=0",
                    "mode=cache_hit_skip",
                    f"outputs={','.join(str(p) for p in outs)}",
                ]
            ),
            encoding="utf-8",
        )
        print(stage_meta)
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_log = LOG_DIR / f"{stage}_{ts}.log"
    stage_nb = NB_DIR / f"{stage}_{ts}.ipynb"
    stage_meta = META_DIR / f"{stage}_{ts}.txt"

    env = os.environ.copy()
    env.update(FAST_PROFILE)
    env["FIG_OUT_DIR"] = str(FIG_DIR)
    env["TABLE_OUT_DIR"] = str(TABLE_DIR)
    env["FIG_DATA_OUT_DIR"] = str(TABLE_DIR)

    # Stage gates: run one heavy stage at a time for stability.
    env["FIG6_ENABLE"] = "1" if stage == "fig6" else "0"
    env["FIG7_ENABLE"] = "1" if stage == "fig7" else "0"
    env["FIG8_ENABLE"] = "1" if stage == "fig8" else "0"
    env["TABLE1_ENABLE"] = "1" if stage == "table1" else "0"

    # core = disable all heavy post-analysis sections
    if stage == "core":
        env["FIG6_ENABLE"] = "0"
        env["FIG7_ENABLE"] = "0"
        env["FIG8_ENABLE"] = "0"
        env["TABLE1_ENABLE"] = "0"

    cmd = ["bash", str(RUNNER)]
    with stage_log.open("w", encoding="utf-8") as f:
        f.write(f"[STAGE] {stage}\n")
        for k in ["CPU_THREADS", "DATALOADER_NUM_WORKERS", "PRED_BATCH_SIZE", "USE_FP16_INFER", "FIG6_ENABLE", "FIG7_ENABLE", "FIG8_ENABLE", "TABLE1_ENABLE"]:
            f.write(f"{k}={env.get(k)}\n")
        f.write("\n")
        rc = subprocess.run(cmd, cwd=BASE_DIR, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

    # Archive final notebook snapshot for this stage.
    src_nb = BASE_DIR / "final_improve_output.ipynb"
    if src_nb.exists():
        shutil.copy2(src_nb, stage_nb)

    stage_meta.write_text(
        "\n".join(
            [
                f"stage={stage}",
                f"return_code={rc}",
                f"log={stage_log}",
                f"snapshot_notebook={stage_nb if src_nb.exists() else 'N/A'}",
            ]
        ),
        encoding="utf-8",
    )
    print(stage_meta)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

