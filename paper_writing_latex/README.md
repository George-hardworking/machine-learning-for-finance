# NeurIPS 2026 Paper: CNN Price-Trend Reproduction

## Project Overview

This directory contains the **NeurIPS 2026 LaTeX paper** that documents the reproduction study of Jiang, Kelly, and Xiu (2023): *"Re-Imagining Price Trends."* The paper presents an independent re-implementation of the image-based price-trend prediction framework using CRSP daily U.S. equity data (1992–2024), with horizon-specific CNN ensembles, factor-spanning regressions, and saliency analysis.

## Paper Info

- **Title**: Re-Implementing CNN Price-Trend Signals: A Reproduction of Jiang, Kelly, and Xiu (2023)
- **Author**: Kaibiao Zhu (HKUST-GZ)
- **Venue**: NeurIPS 2026 (preprint format)
- **Compiled PDF**: `neurips_2026.pdf`

## Repository Structure

```text
paper_writing_latex/
├── neurips_2026.tex                # Main paper source
├── neurips_2026.pdf                # Compiled paper
├── neurips_2026.sty                # NeurIPS style file
├── references.bib                  # BibTeX bibliography
├── checklist.tex                   # NeurIPS submission checklist
├── figures/                        # Paper figures (PDF/PNG)
├── scripts/
│   └── stage_figures.sh            # Figure staging script
└── *.aux, *.bbl, *.blg, *.log, *.out  # LaTeX build artifacts
```

## Paper Structure

1. **Introduction** — Related work on chart patterns, deep learning in finance, and model interpretability
2. **Data and Image Construction** — OHLC rendering with moving averages and volume sub-panels
3. **CNN Architecture and Training** — Horizon-specific architectures (5/20/60-day) with training details
4. **Empirical Results** — Decile analysis, portfolio statistics, Fama-French factor spanning regressions
5. **Saliency Visualization** — Input-gradient maps showing which chart regions drive predictions
6. **Discussion** — Differences from the original paper and implications

## How to Compile

Compile with `pdflatex` + `bibtex`:

```bash
pdflatex neurips_2026.tex
bibtex neurips_2026
pdflatex neurips_2026.tex
pdflatex neurips_2026.tex
```

Or use `latexmk`:

```bash
latexmk -pdf neurips_2026.tex
```

## Figure Staging

To copy the latest figures from `final_submit_version/outputs/pipeline_runs/figures/` into the `figures/` directory:

```bash
bash scripts/stage_figures.sh
```

## Dependencies

- LaTeX distribution (TeX Live / MacTeX) with NeurIPS 2026 style
- `neurips_2026.sty` (included in this directory)

## License

This paper is for course submission and academic use.
