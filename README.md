# STA2453 – Stellar Flare Detection with Isolation Forests

This repository contains a term project for STA2453 (WINTER 2025) at the University of Toronto, Department of Statistical Sciences. The accompanying [`report.pdf`](./report.pdf) outlines the work on detecting stellar flares in TESS light curve data using unsupervised anomaly detection techniques and classical time series analysis. Synthetic flares are simulated and injected into a real lightcurve and used to evaluate the performance of Isolation Forests against a sigma-clipping baseline model.

**Project Manager:** [STA2453 - Project Board](https://voracious-cereal-782.notion.site/1438b211751680248c78eaaaf45572ac?v=4aad22e7653a46b6aa719c4dfa2ed213)

**Style Guide:** [Google Python Style Guide](https://google.github.io/styleguide/pyguide)

---

## Project Overview

**Goal:** Detect stellar flares from a univariate star brightness time series.

**Approach:**
- **Dropped:** Impute missing values with ARIMA forecasting.
- Use STL decomposition to detrend light curves.
- Inject synthetic flares of varied size and shape into the data.
- Detect anomalies using:
  - Isolation Forest (unsupervised learning: anomaly detection model)
  - Sigma-clipping (baseline method)
- Evaluate using event-level precision, recall, and F1 scores.

---

## Repository Structure

- `report.pdf`: walks through all of the work done this term.
- `data` folder:
    - `raw` subfolder: raw data
    - `processed` subfolder: imputed and example flare-injected datasets
- `code` folder:
    - `rough` subfolder: all rough work
    - **Python Notebooks** (`.ipynb`) are where all exploratory work was performed, utilizing their modular nature.
        - `preprocessing`: final attempt at missing value imputation via ARIMA modelling with Fourier Series terms to model seasonal patterns (prior attempts kept under `code/rough/preprocessing`).
        - `isolation_forest`: initial exploration of Isolation Forest model.
        - `flare_simulation`: tests for `simulate_flare.py` module.
        - `model_evaluation`: tests for `evaluate_flare`, `stl_isoforest`, and `sigma_clip` modules.
        - `simulations`: repeated injection-recovery experiments involving (1) hyperparameter tuning and (2) model comparisons (simulation notes kept in `tuning_log.txt`).
    - **Python Modules** (`.py`) are where all supporting functions are stored, to be imported into the Notebooks.
        - `stl_isoforest`: combining STL decomposition + Isolation Forest approach into a single function, `STLIF`.
        - `simulate_flare`: functions for generating flares, i.e. `kepler_flare` and all its supporting functions.
        - `evaluate_flare`: functions for calculating flare classification metrics, `event_level_scores` and all its supporting functions.
        - `sigma_clip`: implementation of the Sigma-clipping method, `sigma_clip`.
- `results` folder: course work - proposal, exploratory data analysis (eda), progress report, and figures.

