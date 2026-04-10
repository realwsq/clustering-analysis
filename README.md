# Rarely Categorical, Highly Separable Representations Along the Cortical Hierarchy

Supplementary code for the paper:

**"Rarely categorical, highly separable representations along the cortical hierarchy"**
L. Posani\*, S. Wang\*, S. Muscinalli, L. Paninski$, and S. Fusi$ (2026)

\*Equal contribution  $Co-senior authors


## Overview

This repository contains the code used to produce **the clustering analysis of single-neuron response profiles** reported in the paper. 

The method, notation, and parameter names used throughout the code follow the conventions established in the main text and Methods section of the paper. Please refer to the paper for detailed definitions and further context.

Related resources:
- **IBL data preprocessing and RRR training** — [realwsq/brainwide-RRR-encoding-model](https://github.com/realwsq/brainwide-RRR-encoding-model)
- **Decoding and dimensionality analyses** — [lposani/decodanda](https://github.com/lposani/decodanda)

## Getting Started
 
This repository uses [Git LFS](https://git-lfs.com) to track a pre-trained reduced-rank regression (RRR) model used for single-neuron selectivity analysis. Install Git LFS before cloning:
```bash
git lfs install
git clone <repository-url>
```
 
If you have already cloned the repository without Git LFS, run `git lfs pull` to fetch the model file.

### Dependencies

Python 3.11 is required. Install the following packages:
```
numpy>=2.2
pandas>=2.2
scipy>=1.15
scikit-learn>=1.6
matplotlib>=3.10
seaborn>=0.13
tqdm>=4.67
```

Install via pip:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm
```

## Repository Structure

- **Figure scripts**: Files are organized and named according to the corresponding figures in the paper. Each script reproduces the analyses for its respective figure.
- **Helper modules**: Files not named after figures contain supporting utilities and shared functions used across the analyses.


## How to run

Each figure script must be run from **within its own folder**, because all data paths are relative to the script's location (e.g., `../data/`). Output figures and intermediate results are saved automatically to subfolders within each script's directory (e.g., `single_area/results_fig3d/`, `multi_area/results_fig3f/`).

**Single-area figures** (e.g.: `Fig3b-d`, `Ext. Data. Fig6`):
```bash
cd single_area/
python Fig3bcd.py
python Suppl_Fig6b.py
# etc.
```

**Multi-area figures** (e.g.: `Fig3e-f`, `Ext. Data. Fig7`):
```bash
cd multi_area_clustering_analysis/
python Fig3ef.py
```

## Contact

For questions, please contact [shuqi.wang@epfl.ch](mailto:shuqi.wang@epfl.ch).


