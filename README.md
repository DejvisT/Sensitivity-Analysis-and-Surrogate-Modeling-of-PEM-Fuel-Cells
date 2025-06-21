# 🔬 Sensitivity Analysis and Surrogate Modeling of PEM Fuel Cells

This repository contains the full pipeline for performing sensitivity analysis and building surrogate models of the **AlphaPEM** Proton Exchange Membrane (PEM) fuel cell model.

The goal is to:
- Identify and quantify the impact of key operational variables using sensitivity analysis.
- Develop a reduced, interpretable surrogate model (via symbolic regression or ML) that approximates the AlphaPEM model behavior with much lower computational cost.

---

## 📁 Project Structure

```text
pem-fuelcell-surrogate/
│
├── data/                          # Data used throughout the project
│   ├── raw/                       # Raw sampled data from AlphaPEM (CSV files)
│   ├── processed/                 # Cleaned/transformed datasets
│   └── external/                  # Any external datasets (e.g. from papers)
│
├── notebooks/                     # Jupyter notebooks for step-by-step analysis
│   ├── 01_exploration.ipynb       # Initial EDA of AlphaPEM outputs
│   ├── 02_sensitivity_analysis.ipynb
│   ├── 03_symbolic_regression.ipynb
│   ├── 04_ml_surrogate_models.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/                           # Core source code
│   ├── __init__.py
│   ├── config.py                  # Project-wide constants and configuration
│   ├── sampling/
│   │   └── sampler.py             # AlphaPEM input space sampler
│   ├── analysis/
│   │   ├── sensitivity.py         # Sensitivity analysis tools
│   │   ├── symbolic_regression.py # Wrapper for PySR or other tools
│   │   ├── surrogate_models.py    # ML models: random forest, GP, etc.
│   │   └── evaluation.py          # Evaluation metrics and comparison plots
│   └── utils/
│       └── io.py                  # CSV I/O, logging, plotting utilities
│
├── models/
│   ├── pysr_equations/            # Symbolic expressions learned from PySR
│   └── trained_ml_models/         # Pickled scikit-learn models or similar
│
├── tests/                         # Unit tests
│   ├── test_sensitivity.py
│   ├── test_sampling.py
│   └── test_surrogate.py
│
├── scripts/                       # CLI scripts for running pipeline components
│   ├── run_sampling.py            # CLI script for generating samples from AlphaPEM
│   ├── run_analysis.py            # Run full sensitivity analysis pipeline
│   └── run_surrogate.py           # Train and evaluate surrogate models
│
├── requirements.txt               # Python dependencies
├── environment.yml                # Optional: conda environment
└── README.md                      # Project overview and how to run

```
## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/DejvisT/Sensitivity-Analysis-and-Surrogate-Modeling-of-PEM-Fuel-Cells.git
```

### 2. Navigate to project directory

```bash
cd Sensitivity-Analysis-and-Surrogate-Modeling-of-PEM-Fuel-Cells
```

### 3. Install the required dependencies (eventually in a specific environment):

```bash
pip install requirements.txt
```

### 4. Install AlphaPEM:
```bash
cd external
git clone https://github.com/gassraphael/AlphaPEM.git
cd AlphaPEM
git checkout 2b042c3
```
