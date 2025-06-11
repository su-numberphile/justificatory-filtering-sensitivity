# Sensitivity Analysis of Justificatory Filtering

This repository contains the code and outputs for a Monte Carlo sensitivity analysis of the inequality:

    (1 - q)(γ + β) > δ

This inequality captures the conditions under which applying a justificatory filter to punishment is more cost-effective than indiscriminate sanctioning. The simulation is part of a project on the evolution and ethics of punishment.

## Files

- `sensitivity_analysis.py` — Main Python script for the simulation and plotting.
- `sensitivity_analysis_output.csv` — CSV output with simulation results.
- `filtering_efficiency_plot.png` — Visualization of filtering efficiency.
- `parameter_distributions_by_outcome.png` — Distributions of key parameters by outcome.

## Requirements

Python 3.7+ with the following packages:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Install requirements using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

To run the analysis:

```bash
python sensitivity_analysis.py
```

This will generate:
- Summary statistics printed to the console
- Two plots saved as `.png` files
- A CSV file with all simulation data

## Author

Ivan Gonzalez-Cabrera  
University of Konstanz  
Email: idgonzalezc@gmail.com