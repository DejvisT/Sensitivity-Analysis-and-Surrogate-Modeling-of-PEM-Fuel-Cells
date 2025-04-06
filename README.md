# Sensitivity Analysis and Surrogate Modeling of PEM Fuel Cells

pem-fuelcell-surrogate/
│
├── data/
│   ├── raw/                       # Raw sampled data from AlphaPEM (CSV files)
│   ├── processed/                 # Cleaned/transformed datasets
│   └── external/                  # Any external datasets (e.g. from papers)
│
├── notebooks/
│   ├── 01_exploration.ipynb       # Initial EDA of AlphaPEM outputs
│   ├── 02_sensitivity_analysis.ipynb
│   ├── 03_symbolic_regression.ipynb
│   ├── 04_ml_surrogate_models.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
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
├── tests/
│   ├── test_sensitivity.py
│   ├── test_sampling.py
│   └── test_surrogate.py
│
├── scripts/
│   ├── run_sampling.py            # CLI script for generating samples from AlphaPEM
│   ├── run_analysis.py            # Run full sensitivity analysis pipeline
│   └── run_surrogate.py           # Train and evaluate surrogate models
│
├── requirements.txt               # Python dependencies
├── environment.yml                # Optional: conda environment
└── README.md                      # Project overview and how to run
 
