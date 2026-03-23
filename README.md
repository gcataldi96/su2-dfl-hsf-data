# Data and code for "Disorder-Free Localization and Fragmentation in a Non-Abelian Lattice Gauge Theory"

This repository contains the datasets and Python scripts used to reproduce the figures in the paper.

The canonical datasets are stored as compressed NumPy archives in `data/`.

## Quickstart
1. Create an environment.
2. Install the Python dependencies.
3. Generate the figures.

The simplest setup is a plain Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python make_all_figures.py
```

If you prefer Conda, an optional environment file is included:

```bash
conda env create -f environment.yml
conda activate su2-dfl
python make_all_figures.py
```

This writes the figures into `figures/`.

## Repository Scripts

- `data_analysis_MAIN.py`: generates the main-text figures.
- `data_analysis_SM.py`: generates the supplemental figures.
- `make_all_figures.py`: runs both figure scripts end-to-end using the committed `data/*.npz` files.

## Data Files

Dataset names are kept semantic rather than figure-number-based. The current plotting workflow uses:

- `phase_diagram_bg.npz`
- `phase_diagram_nobg.npz`
- `dynamics_dfl.npz`
- `frag_imbalance.npz`
- `population_scan_bg.npz`
- `population_scan_nobg.npz`
- `entropy_bg.npz`
- `entropy_nobg.npz`
- `frag_spectrum.npz`
- `frag_dynamics.npz`
- `effective_model_comparison.npz`
- `qmbs_vs_hsf_l8.npz`

## Dependencies

Python packages:

- `numpy`
- `scipy`
- `matplotlib`

The plotting scripts generate PDF figures directly.
