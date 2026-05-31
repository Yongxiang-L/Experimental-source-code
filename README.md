# Experimental Source Code for "Multi‑Product Multi‑Period Blood Inventory Management Integrating Demand Correlation"

This repository contains the complete source code to reproduce the numerical experiments reported in the paper submitted to *Computers & Operations Research*.

The paper proposes a two‑stage robust optimization (TSRO) model for hospital blood inventory management, a data‑driven minimum volume covering ellipsoid (MVCE) uncertainty set that captures cross‑product demand correlations, and an improved column‑and‑constraint generation (I‑C&CG) algorithm. The code reproduces all tables and figures from **Section 5** (Numerical Experiments).

## Repository Structure

```
.
├── 5.1. Effectiveness of the Proposed Algorithm.py          # Table 3 (I-C&CG vs. classical C&CG)
├── 5.1.3. Ablation Study Impact of the Multi-Start and GA Mechanisms.py  # Table 4
├── 5.2. Compactness Value MVCE Uncertainty Set vs. Polyhedral Uncertainty Set.py  # Table 5
├── 5.3. Impact of Demand Correlation.py                     # Table 6
├── 5.4. Impact of Shelf Life.py                             # Table 7
└── README.md
```

## Requirements

- Python 3.11.4 (or later 3.x)
- [Gurobi Optimizer](https://www.gurobi.com/) 12.0.3 with a valid license
- Python packages: `numpy`, `scipy`, `pandas`, `gurobipy` (installed with the Gurobi package)

All required packages can be installed via:

```bash
pip install numpy scipy pandas gurobipy
```

> **Note**: `gurobipy` is included in the Gurobi installation. After installing Gurobi, run `python -m pip install gurobipy` or use the Gurobi shell command.

## How to Run

Each script is self‑contained and reproduces a specific part of Section 5. Run any script directly from the command line, for example:

```bash
python "5.1. Effectiveness of the Proposed Algorithm.py"
```

The scripts will:

1. Generate synthetic historical demand data using a truncated multivariate normal distribution (as described in Section 5.1.1).
2. Calibrate the MVCE uncertainty set with the Khachiyan algorithm (Algorithm 1 in the paper).
3. Run the proposed I‑C&CG algorithm (and, where applicable, the classical C&CG or its variants).
4. Perform Monte Carlo out‑of‑sample simulations (10 000 scenarios, depending on the experiment) to evaluate the obtained procurement plans.
5. Print summary tables (e.g., Table 3, Table 5) to the console and save detailed results in CSV files under the corresponding output directory (`./Algorithm_Comparison_Results/`, `./ablation_results/`, `./5.2_Experiment_Output_By_Scale/`, `./5.3 experimental result/`, `./5.4_Experiment_Results/`).

> **Important**: The random seed is fixed inside each script (e.g., `np.random.seed(20260401)`). Therefore, running the same script multiple times should produce identical results, ensuring full reproducibility.

## Experiment Descriptions and Corresponding Paper Sections

| File | Paper Section | Description |
|------|---------------|-------------|
| `5.1. Effectiveness of the Proposed Algorithm.py` | 5.1, Table 3 | Compares the proposed I‑C&CG algorithm with the classical C&CG algorithm for problem scales B = {2,3} and T = {3,5,7}. Reports number of iterations, solution time, optimal cost, and the average relative difference ratio. |
| `5.1.3. Ablation Study Impact of the Multi-Start and GA Mechanisms.py` | 5.1.3, Table 4 | Quantifies the individual contributions of the multi‑start initialization and the gradient‑ascent inner loop. Three variants are compared: full I‑C&CG, without multi‑start, and without gradient ascent (no GA). |
| `5.2. Compactness Value MVCE Uncertainty Set vs. Polyhedral Uncertainty Set.py` | 5.2, Table 5 | Compares the proposed MVCE uncertainty set with the classical polyhedral (budgeted) uncertainty set. Both are calibrated to cover 100% of historical samples. The script evaluates first‑stage cost, holding cost, shortage cost, expiration cost, total cost, and stockout rate via Monte Carlo simulation. |
| `5.3. Impact of Demand Correlation.py` | 5.3, Table 6 | Varies the Pearson correlation coefficient ρ from –0.5 to 0.8 (fixed B=3, T=7, σ=0.5). Shows how demand correlation affects the optimal total cost, procurement cost, holding cost, shortage cost, expiration cost, and stockout rate. |
| `5.4. Impact of Shelf Life.py` | 5.4, Table 7 | Varies the maximum shelf life of blood products M<sub>b</sub> from 3 to 8 days (B=3, T=7, ρ=0.3, σ=0.5). Demonstrates the relationship between shelf life and total cost, and the trade‑off between expiration risk and stockout risk. |

## Output Files

Each script creates an output folder (if it does not already exist) and saves:

- **Detailed raw results** for every repetition (e.g., `algorithm_comparison_raw.csv`).
- **Averaged summary statistics** over repetitions (e.g., `algorithm_comparison_summary.csv`).
- In the case of the ablation study, the summary is printed to the console and also saved as `ablation_results.csv`.

The naming of output files follows the pattern `*_Experiment_*.csv` so that you can easily identify which experiment produced them.

## Reproducing the Tables

To exactly reproduce the numbers shown in the paper tables (e.g., Tables 3–7), simply run the corresponding Python script. Because all stochastic components are seeded, the results should match the reported values up to small numerical tolerances.

If you wish to change the problem scale (e.g., B=5 for Table 3 or T=10), you can modify the lists `B_list` and `T_list` inside the `EXPERIMENT_PARAMS` dictionary at the top of each script.

## Contact

For questions about the code or the paper, please contact the authors:

**Yongxiang Liang**   
[yongxiang.liang@mail.sdu.edu.cn]

**Pengcheng Dong**  
[iedpc@mail.sdu.edu.cn]

