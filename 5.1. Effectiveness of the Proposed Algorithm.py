import os
import numpy as np
from scipy.stats import multivariate_normal
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd
from typing import Tuple, List, Dict
from joblib import Parallel, delayed

# ==============================================================================
# ===================== Part 1: Global Parameters and Common Utilities =====================
# ==============================================================================

np.random.seed(20260401)  # Fix random seed for full reproducibility of experimental results

# Base cost parameters for all blood product categories (per unit)
BASE_COSTS = {
    "CVb": [200, 200, 200],  # Unit variable procurement cost
    "CHb": [10, 10, 10],  # Unit holding cost per period
    "CSb": [2000, 2000, 2000],  # Unit shortage cost
    "COb": [200, 200, 200]  # Unit expiration disposal cost
}

# Experimental parameters strictly following Section 5.1.1 of the manuscript
EXPERIMENT_PARAMS = {
    "CF": 200,  # Fixed ordering cost per batch
    "epsilon": 1e-6,  # Convergence tolerance as specified in the paper
    "max_outer_iter": 200,  # Maximum number of outer iterations
    "time_limit": 3600,  # Time limit per solver run (3600 seconds)
    "M_big": 1e4,  # Sufficiently large constant for big-M constraints
    "B_list": [2, 3, 5],  # Number of blood product categories tested
    "T_list": [3, 5, 7],  # Planning horizon lengths tested (days)
    "n_history": 365,  # One year of historical demand data for uncertainty set calibration
    "n_repeat": 100,  # Number of independent replications per experiment
    "rho": 0.3,  # Pearson correlation coefficient between cross-product demands
    "output_dir": "./Algorithm_Comparison_Results",
    "n_jobs": 1,  # Sequential execution to eliminate parallelism-induced variability
    "gurobi_threads": 1,  # Single-threaded execution for accurate timing measurements
    "gurobi_outputflag": 0,  # Disable Gurobi solver output by default
    "Mb": 5,  # Maximum shelf life of blood products (5 days)
    "demand_upper_bound_ratio": 5.0,
    "max_consecutive_duplicate": 3,  # Terminate if no new scenarios are found for 3 consecutive iterations
    # I-C&CG algorithm specific parameters
    "bcd_inner_epsilon": 1e-4,  # Inner loop convergence tolerance for gradient ascent
    "bcd_max_inner_iter": 100,  # Maximum inner loop iterations
    "grad_epsilon": 1e-4,  # Numerical tolerance for gradient calculation
    "n_multi_cut_scenarios": 3,  # Number of cutting planes added per outer iteration
    "debug_infeasibility": False,
}


def get_cost_params(B: int) -> dict:
    """Retrieve cost parameters for a given number of blood product categories."""
    return {
        "CF": EXPERIMENT_PARAMS["CF"],
        "CVb": np.array(BASE_COSTS["CVb"][:B]),
        "CHb": np.array(BASE_COSTS["CHb"][:B]),
        "CSb": np.array(BASE_COSTS["CSb"][:B]),
        "COb": np.array(BASE_COSTS["COb"][:B]),
        "Mb": EXPERIMENT_PARAMS["Mb"],
        "M_big": EXPERIMENT_PARAMS["M_big"]
    }


def set_gurobi_params(model: gp.Model, is_subproblem: bool = False, force_output: bool = False):
    """
    Configure Gurobi solver parameters for optimal performance and numerical stability.
    Args:
        model: Gurobi model instance
        is_subproblem: Whether the model is a subproblem (enforces stricter numerical settings)
        force_output: Override output flag to show solver logs
    """
    model.Params.Threads = EXPERIMENT_PARAMS["gurobi_threads"]
    model.Params.OutputFlag = 1 if force_output else EXPERIMENT_PARAMS["gurobi_outputflag"]
    model.Params.TimeLimit = EXPERIMENT_PARAMS["time_limit"]
    model.Params.Method = -1  # Automatic method selection
    model.Params.MIPFocus = 1  # Focus on finding feasible solutions quickly
    model.Params.Cuts = 1  # Moderate cut generation
    model.Params.DualReductions = 0  # Disable dual reductions for numerical stability
    model.Params.PreDual = 0
    model.Params.PreQLinearize = 1  # Linearize quadratic terms in presolve
    model.Params.Presolve = 1  # Standard presolve
    model.Params.FeasibilityTol = 1e-4
    model.Params.OptimalityTol = 1e-4
    model.Params.IntFeasTol = 1e-4
    model.Params.InfUnbdInfo = 1  # Enable infeasibility/unboundedness information
    model.Params.NumericFocus = 2  # Moderate numerical focus

    if is_subproblem:
        model.Params.NumericFocus = 3  # Higher numerical focus for subproblems
        model.Params.NonConvex = 2  # Allow non-convex quadratic constraints
        model.Params.MIPGap = 1e-4
    else:
        model.Params.NumericFocus = 2
    return model


def khachiyan_algorithm(data: np.ndarray, tol: float = 1e-6, max_iter: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Khachiyan's algorithm to compute the Minimum Volume Covering Ellipsoid (MVCE) for given data.
    Args:
        data: Input data matrix (n_samples x n_dim)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
    Returns:
        a: Center vector of the MVCE
        H: Shape matrix of the MVCE (H = (B * Σ)⁻¹)
    """
    n_samples, n_dim = data.shape
    Q = np.vstack([data.T, np.ones(n_samples)])  # Homogeneous transformation
    p = np.ones(n_samples) / n_samples  # Initial uniform weights
    reg = 1e-4 * np.eye(Q.shape[0])  # Regularization to avoid singular matrices

    for _ in range(max_iter):
        Lambda = Q @ np.diag(p) @ Q.T + reg
        try:
            Lambda_inv = np.linalg.inv(Lambda)
        except np.linalg.LinAlgError:
            Lambda += 1e-4 * np.eye(Lambda.shape[0])
            Lambda_inv = np.linalg.inv(Lambda)

        g = np.diag(Q.T @ Lambda_inv @ Q)
        j, g_max = np.argmax(g), g.max()

        if g_max <= (1 + tol) * (n_dim + 1):
            break

        # Update weights to expand the ellipsoid towards the outlier point
        omega = (g_max - n_dim - 1) / ((n_dim + 1) * (g_max - 1)) if g_max > n_dim + 1 else 0
        p = (1 - omega) * p
        p[j] += omega
        p /= np.sum(p)  # Renormalize weights

    # Calculate final MVCE parameters
    a = data.T @ p
    centered_data = data - a
    Sigma = centered_data.T @ np.diag(p) @ centered_data
    H = np.linalg.inv(n_dim * Sigma + 1e-4 * np.eye(n_dim))

    return a, H


def generate_truncated_mvn(B: int, n_samples: int, rho: float = 0.3) -> np.ndarray:
    """
    Generate non-negative truncated multivariate normal demand samples.
    Args:
        B: Number of blood product categories
        n_samples: Number of samples to generate
        rho: Pearson correlation coefficient between cross-product demands
    Returns:
        Non-negative demand matrix (n_samples x B)
    """
    # Mean demand values for different numbers of product categories
    if B == 2:
        mu = np.array([40, 30])
    elif B == 3:
        mu = np.array([40, 35, 30])
    elif B == 5:
        mu = np.array([40, 35, 30, 25, 20])
    else:
        raise ValueError(f"Number of product categories B={B} not supported")

    sigma = 0.5 * mu  # Standard deviation is 50% of the mean for each product
    # Construct covariance matrix with specified correlation structure
    Sigma = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            Sigma[i, j] = sigma[i] ** 2 if i == j else rho * sigma[i] * sigma[j]
    Sigma += 1e-4 * np.eye(B)  # Regularization to ensure positive definiteness

    mvn = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)
    return np.maximum(mvn.rvs(size=n_samples), 0)  # Truncate to non-negative values


# ==============================================================================
# ===================== Part 2: Classical C&CG Algorithm Implementation =====================
# ==============================================================================

def build_classic_ccg_subproblem(x_current: np.ndarray, T: int, B: int, a: np.ndarray, H: np.ndarray,
                                 cost_params: dict):
    """
    Build the subproblem for the classical C&CG algorithm using KKT conditions and big-M linearization.
    Args:
        x_current: Current first-stage order quantity solution from the master problem
        T: Planning horizon length
        B: Number of blood product categories
        a: Center of the MVCE uncertainty set
        H: Shape matrix of the MVCE uncertainty set
        cost_params: Dictionary of cost parameters
    Returns:
        Gurobi model and all decision variables
    """
    CHb, CSb, COb, Mb, M_big = cost_params["CHb"], cost_params["CSb"], cost_params["COb"], cost_params["Mb"], \
    cost_params["M_big"]

    model = gp.Model("Classic_CCG_SP")
    model = set_gurobi_params(model, is_subproblem=True)

    # Decision variables
    d = model.addVars(T, B, lb=0, name="d")  # Demand variables (uncertain parameters)
    i = model.addVars(T, B, Mb, lb=0, name="i")  # Inventory by remaining shelf life
    u = model.addVars(T, B, Mb, lb=0, name="u")  # Usage by remaining shelf life
    s = model.addVars(T, B, lb=0, name="s")  # Shortage variables
    o = model.addVars(T, B, lb=0, name="o")  # Expiration (wastage) variables

    # KKT dual variables
    alpha = model.addVars(T, B, lb=0, name="alpha")
    beta = model.addVars(T, B, Mb, lb=0, name="beta")
    nu = model.addVars(B, Mb - 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="nu")
    gamma = model.addVars(T - 1, B, Mb, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="gamma")
    delta = model.addVars(T, B, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="delta")
    eta = model.addVars(T, B, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="eta")
    pai = model.addVars(T, B, lb=0, name="pai")

    # Dual variables for non-negativity constraints
    theta = model.addVars(T, B, lb=0, name="theta")
    lam = model.addVars(T, B, lb=0, name="lam")
    mu = model.addVars(T, B, Mb, lb=0, name="mu")
    rho_var = model.addVars(T, B, Mb, lb=0, name="rho")

    # Binary variables for complementary slackness conditions (big-M method)
    y_a = model.addVars(T, B, vtype=GRB.BINARY, name="y_a")
    y_b = model.addVars(T, B, Mb, vtype=GRB.BINARY, name="y_b")
    y_c = model.addVars(T, B, vtype=GRB.BINARY, name="y_c")
    y_s = model.addVars(T, B, vtype=GRB.BINARY, name="y_s")
    y_o = model.addVars(T, B, vtype=GRB.BINARY, name="y_o")
    y_i = model.addVars(T, B, Mb, vtype=GRB.BINARY, name="y_i")
    y_u = model.addVars(T, B, Mb, vtype=GRB.BINARY, name="y_u")

    # Objective: Maximize the second-stage recourse cost (worst-case scenario)
    cost_h = gp.quicksum(CHb[b] * i[t, b, m] for t in range(T) for b in range(B) for m in range(Mb))
    cost_s = gp.quicksum(CSb[b] * s[t, b] for t in range(T) for b in range(B))
    cost_o = gp.quicksum(COb[b] * o[t, b] for t in range(T) for b in range(B))
    model.setObjective(cost_h + cost_s + cost_o, GRB.MAXIMIZE)

    # Primal problem constraints
    for t in range(T):
        # MVCE uncertainty set constraint for each period
        expr = 0
        for i1 in range(B):
            for j1 in range(B):
                expr += H[i1, j1] * (d[t, i1] - a[i1]) * (d[t, j1] - a[j1])
        model.addConstr(expr <= 1, name=f"ellipsoid_{t}")

    # Initial inventory: no carryover from before the planning horizon
    for b in range(B):
        for m in range(Mb - 1):
            model.addConstr(i[0, b, m] == 0, name=f"init_i_{b}_{m}")

    for t in range(T):
        for b in range(B):
            # Demand balance: total usage + shortage >= realized demand
            model.addConstr(gp.quicksum(u[t, b, m] for m in range(Mb)) + s[t, b] >= d[t, b],
                            name=f"demand_balance_{t}_{b}")
            # Total usage cannot exceed demand (no over-usage allowed)
            model.addConstr(gp.quicksum(u[t, b, m] for m in range(Mb)) <= d[t, b], name=f"sum_u_le_d_{t}_{b}")

            # Usage from each batch cannot exceed available inventory
            for m in range(Mb):
                model.addConstr(u[t, b, m] <= i[t, b, m], name=f"u_le_i_{t}_{b}_{m}")

            # Newly ordered products have full remaining shelf life
            model.addConstr(i[t, b, Mb - 1] == x_current[t, b], name=f"replenish_{t}_{b}")
            # Expiration: unconsumed inventory with 1 day remaining expires at end of period
            model.addConstr(o[t, b] == i[t, b, 0] - u[t, b, 0], name=f"expire_{t}_{b}")

    # Inventory carryover: remaining shelf life decreases by 1 each period
    for t in range(T - 1):
        for b in range(B):
            for m in range(1, Mb):
                model.addConstr(i[t + 1, b, m - 1] == i[t, b, m] - u[t, b, m], name=f"shift_{t}_{b}_{m}")

    # KKT stationarity conditions
    for t in range(T):
        for b in range(B):
            model.addConstr(CSb[b] - alpha[t, b] - theta[t, b] == 0, name=f"KKT_s_{t}_{b}")
            model.addConstr(COb[b] + eta[t, b] - lam[t, b] == 0, name=f"KKT_o_{t}_{b}")
            model.addConstr(-alpha[t, b] + beta[t, b, 0] + eta[t, b] + pai[t, b] - rho_var[t, b, 0] == 0,
                            name=f"KKT_u1_{t}_{b}_0")

            if t <= T - 2:
                for m in range(1, Mb):
                    model.addConstr(-alpha[t, b] + beta[t, b, m] + gamma[t, b, m] + pai[t, b] - rho_var[t, b, m] == 0,
                                    name=f"KKT_u2_{t}_{b}_{m}")

            if t == T - 1:
                for m in range(1, Mb):
                    model.addConstr(-alpha[t, b] + beta[t, b, m] + pai[t, b] - rho_var[t, b, m] == 0,
                                    name=f"KKT_u3_{t}_{b}_{m}")

    for b in range(B):
        model.addConstr(CHb[b] + nu[b, 0] - beta[0, b, 0] - eta[0, b] - mu[0, b, 0] == 0, name=f"KKT_i_0_{b}_0")

        for m in range(1, Mb - 1):
            model.addConstr(CHb[b] + nu[b, m] - beta[0, b, m] - gamma[0, b, m] - mu[0, b, m] == 0,
                            name=f"KKT_i_0_{b}_{m}")

        model.addConstr(CHb[b] - beta[0, b, Mb - 1] - gamma[0, b, Mb - 1] + delta[0, b] - mu[0, b, Mb - 1] == 0,
                        name=f"KKT_i_0_{b}_{Mb - 1}")

        for t in range(1, T - 1):
            model.addConstr(CHb[b] - beta[t, b, 0] + gamma[t - 1, b, 1] - eta[t, b] - mu[t, b, 0] == 0,
                            name=f"KKT_i_{t}_{b}_0")

            for m in range(1, Mb - 1):
                model.addConstr(CHb[b] - beta[t, b, m] - gamma[t, b, m] + gamma[t - 1, b, m + 1] - mu[t, b, m] == 0,
                                name=f"KKT_i_{t}_{b}_{m}")

            model.addConstr(CHb[b] - beta[t, b, Mb - 1] - gamma[t, b, Mb - 1] + delta[t, b] - mu[t, b, Mb - 1] == 0,
                            name=f"KKT_i_{t}_{b}_{Mb - 1}")

        t = T - 1
        model.addConstr(CHb[b] - beta[t, b, 0] + gamma[t - 1, b, 1] - eta[t, b] - mu[t, b, 0] == 0,
                        name=f"KKT_i_{t}_{b}_0")

        for m in range(1, Mb - 1):
            model.addConstr(CHb[b] - beta[t, b, m] + gamma[t - 1, b, m + 1] - mu[t, b, m] == 0,
                            name=f"KKT_i_{t}_{b}_{m}")

        model.addConstr(CHb[b] - beta[t, b, Mb - 1] + delta[t, b] - mu[t, b, Mb - 1] == 0,
                        name=f"KKT_i_{t}_{b}_{Mb - 1}")

    # Complementary slackness conditions using big-M method
    M = M_big
    for t in range(T):
        for b in range(B):
            slk_a = gp.quicksum(u[t, b, m] for m in range(Mb)) + s[t, b] - d[t, b]
            model.addConstr(alpha[t, b] <= M * y_a[t, b], name=f"CS_a1_{t}_{b}")
            model.addConstr(slk_a <= M * (1 - y_a[t, b]), name=f"CS_a2_{t}_{b}")

            for m in range(Mb):
                slk_b = i[t, b, m] - u[t, b, m]
                model.addConstr(beta[t, b, m] <= M * y_b[t, b, m], name=f"CS_b1_{t}_{b}_{m}")
                model.addConstr(slk_b <= M * (1 - y_b[t, b, m]), name=f"CS_b2_{t}_{b}_{m}")

            slk_c = d[t, b] - gp.quicksum(u[t, b, m] for m in range(Mb))
            model.addConstr(pai[t, b] <= M * y_c[t, b], name=f"CS_c1_{t}_{b}")
            model.addConstr(slk_c <= M * (1 - y_c[t, b]), name=f"CS_c2_{t}_{b}")

            model.addConstr(theta[t, b] <= M * y_s[t, b], name=f"CS_s1_{t}_{b}")
            model.addConstr(s[t, b] <= M * (1 - y_s[t, b]), name=f"CS_s2_{t}_{b}")

            model.addConstr(lam[t, b] <= M * y_o[t, b], name=f"CS_o1_{t}_{b}")
            model.addConstr(o[t, b] <= M * (1 - y_o[t, b]), name=f"CS_o2_{t}_{b}")

            for m in range(Mb):
                model.addConstr(mu[t, b, m] <= M * y_i[t, b, m], name=f"CS_i1_{t}_{b}_{m}")
                model.addConstr(i[t, b, m] <= M * (1 - y_i[t, b, m]), name=f"CS_i2_{t}_{b}_{m}")

            for m in range(Mb):
                model.addConstr(rho_var[t, b, m] <= M * y_u[t, b, m], name=f"CS_u1_{t}_{b}_{m}")
                model.addConstr(u[t, b, m] <= M * (1 - y_u[t, b, m]), name=f"CS_u2_{t}_{b}_{m}")

    model.update()
    return model, d, i, s, o, u


def solve_master_problem(T: int, B: int, scenarios: List[np.ndarray], cost_params: dict):
    """
    Solve the master problem for the classical C&CG algorithm.
    Args:
        scenarios: List of worst-case demand scenarios generated so far
    Returns:
        Gurobi model, optimal order quantities x, binary ordering indicators z, and objective value
    """
    CF, CVb, CHb, CSb, COb, Mb, M_big = cost_params["CF"], cost_params["CVb"], cost_params["CHb"], cost_params["CSb"], \
    cost_params["COb"], cost_params["Mb"], cost_params["M_big"]

    model = gp.Model("Classic_CCG_MP")
    model = set_gurobi_params(model)

    # First-stage decision variables
    x = model.addVars(T, B, lb=0, name="x")  # Order quantities
    z = model.addVars(T, vtype=GRB.BINARY, name="z")  # Ordering indicators
    theta = model.addVar(lb=0, name="theta")  # Auxiliary variable for worst-case second-stage cost

    # Objective: Minimize total first-stage cost + worst-case second-stage cost
    model.setObjective(
        gp.quicksum(CF * z[t] for t in range(T)) +
        gp.quicksum(CVb[b] * x[t, b] for t in range(T) for b in range(B)) +
        theta,
        GRB.MINIMIZE
    )

    # Fixed cost constraint: incur fixed cost only if any product is ordered in the period
    for t in range(T):
        for b in range(B):
            model.addConstr(x[t, b] <= M_big * z[t], name=f"fixed_cost_{t}_{b}")

    num_scenarios = len(scenarios)
    i = {}  # Second-stage inventory variables for each scenario
    u = {}  # Second-stage usage variables for each scenario
    s = {}  # Second-stage shortage variables for each scenario
    o = {}  # Second-stage expiration variables for each scenario

    for j in range(num_scenarios):
        d_j = scenarios[j]
        i[j] = model.addVars(T + 1, B, Mb, lb=0, name=f"i_{j}")
        u[j] = model.addVars(T, B, Mb, lb=0, name=f"u_{j}")
        s[j] = model.addVars(T, B, lb=0, name=f"s_{j}")
        o[j] = model.addVars(T, B, lb=0, name=f"o_{j}")

        # Cutting plane constraint: theta >= second-stage cost for scenario j
        scene_cost_hold = gp.quicksum(CHb[b] * i[j][t, b, m] for t in range(T) for b in range(B) for m in range(Mb))
        scene_cost_stockout = gp.quicksum(CSb[b] * s[j][t, b] for t in range(T) for b in range(B))
        scene_cost_expire = gp.quicksum(COb[b] * o[j][t, b] for t in range(T) for b in range(B))
        model.addConstr(theta >= scene_cost_hold + scene_cost_stockout + scene_cost_expire, name=f"cut_plane_{j}")

        # Initial inventory constraint
        for b in range(B):
            for m in range(Mb - 1):
                model.addConstr(i[j][0, b, m] == 0, name=f"init_i_{j}_{b}_{m}")

        for t in range(T):
            for b_idx in range(B):
                # Replenishment: new orders have full shelf life
                model.addConstr(i[j][t, b_idx, Mb - 1] == x[t, b_idx], name=f"replenish_{j}_{t}_{b_idx}")

                # Demand balance constraint
                model.addConstr(
                    gp.quicksum(u[j][t, b_idx, m] for m in range(Mb)) + s[j][t, b_idx] >= d_j[t, b_idx],
                    name=f"demand_balance_{j}_{t}_{b_idx}"
                )
                # Total usage cannot exceed demand
                model.addConstr(gp.quicksum(u[j][t, b_idx, m] for m in range(Mb)) <= d_j[t, b_idx],
                                name=f"sum_u_le_d_{j}_{t}_{b_idx}")

                # Usage cannot exceed available inventory
                for m in range(Mb):
                    model.addConstr(u[j][t, b_idx, m] <= i[j][t, b_idx, m], name=f"inv_upper_{j}_{t}_{b_idx}_{m}")

                # Expiration calculation
                model.addConstr(o[j][t, b_idx] == i[j][t, b_idx, 0] - u[j][t, b_idx, 0], name=f"expire_{j}_{t}_{b_idx}")

                # Inventory carryover
                for m in range(1, Mb):
                    model.addConstr(i[j][t + 1, b_idx, m - 1] == i[j][t, b_idx, m] - u[j][t, b_idx, m],
                                    name=f"shift_{j}_{t}_{b_idx}_{m}")

    model.optimize()

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        x_val = np.array([[x[t, b].X for b in range(B)] for t in range(T)])
        z_val = np.array([1 if z[t].X > 0.5 else 0 for t in range(T)])
        obj_val = model.ObjVal
        return model, x_val, z_val, obj_val
    else:
        return model, None, None, None


def run_classic_ccg_algorithm(T: int, B: int, a: np.ndarray, H: np.ndarray, cost_params: dict) -> Dict:
    """
    Run the complete classical C&CG algorithm for a given problem instance.
    Returns:
        Dictionary of algorithm performance results
    """
    start_time = time.time()
    epsilon = EXPERIMENT_PARAMS["epsilon"]
    max_outer_iter = EXPERIMENT_PARAMS["max_outer_iter"]
    max_consecutive_duplicate = EXPERIMENT_PARAMS["max_consecutive_duplicate"]
    CF, CVb = cost_params["CF"], cost_params["CVb"]

    # Initialize with the center of the uncertainty set as the first scenario
    initial_scenario = np.tile(a, (T, 1))
    D = [initial_scenario]

    LB, UB = -np.inf, np.inf
    iter_count = 0
    converged = False
    x_opt = None
    consecutive_duplicate = 0

    while iter_count < max_outer_iter:
        iter_count += 1

        # Solve master problem to get current first-stage decision
        model_mp, x_current, z_current, MP_obj = solve_master_problem(T, B, D, cost_params)

        if x_current is None: break
        LB = max(LB, MP_obj)  # Update lower bound

        # Solve subproblem to find the worst-case demand scenario
        model_sp, d_vars, i_sp, s_sp, o_sp, u_sp = build_classic_ccg_subproblem(x_current, T, B, a, H, cost_params)
        model_sp.optimize()

        allowed_statuses = [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]
        if model_sp.status not in allowed_statuses or model_sp.SolCount == 0:
            break

        # Extract worst-case demand scenario
        d_worst = np.array([[max(d_vars[t, b].X, 0) for b in range(B)] for t in range(T)])

        # Calculate current upper bound
        cost_1st_fixed = np.sum(CF * z_current)
        cost_1st_variable = np.sum(CVb * x_current)
        Q_worst = model_sp.ObjVal
        current_UB = cost_1st_fixed + cost_1st_variable + Q_worst

        if current_UB < UB:
            UB = current_UB
            x_opt = x_current.copy()

        # Check for convergence
        gap = np.abs((UB - LB) / LB) if LB > 1e-6 else np.inf
        if gap <= epsilon:
            converged = True
            break

        # Add new scenario to the master problem if it's not a duplicate
        is_new = not any(np.all(np.abs(d_worst - d_exist) < 1e-4) for d_exist in D)
        if is_new:
            D.append(d_worst)
            consecutive_duplicate = 0
        else:
            consecutive_duplicate += 1
            if consecutive_duplicate >= max_consecutive_duplicate:
                break

    total_time = time.time() - start_time

    return {
        "algorithm": "Classic_CCG",
        "T": T, "B": B,
        "converged": converged,
        "total_time": total_time,
        "iter_count": iter_count,
        "LB": LB, "UB": UB,
        "TC_opt": UB if converged else np.nan,
        "x_opt": x_opt,
        "scenario_count": len(D),
        "time_per_iter": total_time / iter_count if iter_count > 0 else np.nan
    }


# ==============================================================================
# ===================== Part 3: Improved C&CG (I-C&CG) Algorithm Implementation =====================
# ==============================================================================

def sample_ellipsoid_points(a: np.ndarray, H: np.ndarray, n_samples: int) -> List[np.ndarray]:
    """
    Generate uniformly distributed points on the boundary of the MVCE uncertainty set.
    Uses dimension-specific sampling strategies for optimal coverage.
    """
    B = len(a)
    samples = []

    # Cholesky decomposition of the shape matrix for linear transformation
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        reg = 1e-4 * np.eye(B)
        L = np.linalg.cholesky(H + reg)

    if B == 2:
        # 2D: Polar coordinate uniform division for perfect uniformity
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        for theta in angles:
            z = np.array([np.cos(theta), np.sin(theta)])
            y = np.linalg.solve(L.T, z)
            d_sample = a + y
            d_sample = np.maximum(d_sample, 0.0)
            samples.append(d_sample)
    elif B == 3:
        # 3D: Fibonacci sphere sampling for quasi-uniform distribution
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
        for i in range(n_samples):
            y_coord = 1 - (i / float(n_samples - 1)) * 2
            radius = np.sqrt(1 - y_coord * y_coord)
            theta = phi * i
            x_coord = np.cos(theta) * radius
            z_coord = np.sin(theta) * radius
            z = np.array([x_coord, y_coord, z_coord])
            y = np.linalg.solve(L.T, z)
            d_sample = a + y
            d_sample = np.maximum(d_sample, 0.0)
            samples.append(d_sample)
    else:
        # High-dimensional: Low-discrepancy sampling using normalized random vectors
        for i in range(n_samples):
            z = np.random.randn(B)
            norm_z = np.linalg.norm(z)
            if norm_z > 1e-8:
                z /= norm_z
            y = np.linalg.solve(L.T, z)
            d_sample = a + y
            d_sample = np.maximum(d_sample, 0.0)
            samples.append(d_sample)

    return samples


def second_stage_LP_solver(x: np.ndarray, d: np.ndarray, cost_params: dict) -> Tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the second-stage linear programming problem for given first-stage decisions and demand scenario.
    Returns:
        Q: Optimal second-stage cost
        u_val, s_val, o_val, i_val: Optimal values of second-stage variables
    """
    CHb, CSb, COb, Mb = cost_params["CHb"], cost_params["CSb"], cost_params["COb"], cost_params["Mb"]
    T, B = x.shape

    model = gp.Model("SecondStage_LP")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.NumericFocus = 3  # High numerical focus for stability

    i = model.addVars(T + 1, B, Mb, lb=0.0, name="i")
    u = model.addVars(T, B, Mb, lb=0.0, name="u")
    s = model.addVars(T, B, lb=0.0, name="s")
    o = model.addVars(T, B, lb=0.0, name="o")

    # Objective: Minimize total holding + shortage + expiration cost
    total_cost = 0.0
    for t in range(T):
        for b in range(B):
            for m in range(Mb):
                total_cost += CHb[b] * i[t, b, m]
            total_cost += CSb[b] * s[t, b]
            total_cost += COb[b] * o[t, b]
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Initial inventory constraint
    for b in range(B):
        for m in range(Mb - 1):
            model.addConstr(i[0, b, m] == 0.0, name=f"init_i_{b}_{m}")

    for t in range(T):
        for b_idx in range(B):
            # Replenishment: new orders have full shelf life
            model.addConstr(i[t, b_idx, Mb - 1] == x[t, b_idx], name=f"replenish_{t}_{b_idx}")

            # Demand balance
            model.addConstr(
                gp.quicksum(u[t, b_idx, m] for m in range(Mb)) + s[t, b_idx] >= d[t, b_idx],
                name=f"demand_balance_{t}_{b_idx}"
            )
            # Total usage cannot exceed demand
            model.addConstr(
                gp.quicksum(u[t, b_idx, m] for m in range(Mb)) <= d[t, b_idx],
                name=f"sum_u_le_d_{t}_{b_idx}"
            )

            # Usage cannot exceed available inventory
            for m in range(Mb):
                model.addConstr(u[t, b_idx, m] <= i[t, b_idx, m], name=f"inv_upper_{t}_{b_idx}_{m}")

            # Expiration calculation
            model.addConstr(o[t, b_idx] == i[t, b_idx, 0] - u[t, b_idx, 0], name=f"expire_{t}_{b_idx}")

            # Inventory carryover
            for m in range(1, Mb):
                model.addConstr(
                    i[t + 1, b_idx, m - 1] == i[t, b_idx, m] - u[t, b_idx, m],
                    name=f"shift_{t}_{b_idx}_{m}"
                )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        Q = model.ObjVal
        i_val = np.zeros((T + 1, B, Mb), dtype=np.float64)
        u_val = np.zeros((T, B, Mb), dtype=np.float64)
        s_val = np.zeros((T, B), dtype=np.float64)
        o_val = np.zeros((T, B), dtype=np.float64)

        for t in range(T):
            for b in range(B):
                s_val[t, b] = s[t, b].X
                o_val[t, b] = o[t, b].X
                for m in range(Mb):
                    i_val[t, b, m] = i[t, b, m].X
                    u_val[t, b, m] = u[t, b, m].X
        for b in range(B):
            for m in range(Mb):
                i_val[T, b, m] = i[T, b, m].X

        model.dispose()
        return Q, u_val, s_val, o_val, i_val
    else:
        model.dispose()
        raise Exception(f"Second-stage LP solver failed with status code: {model.status}")


def compute_gradient(x: np.ndarray, d: np.ndarray, cost_params: dict) -> np.ndarray:
    """
    Compute the analytical gradient of the second-stage cost function Q with respect to demand d.
    Uses reverse-order loop from last period to first for efficient calculation.
    Returns:
        alpha: Gradient matrix (T x B)
    """
    CHb, CSb, COb, Mb = cost_params["CHb"], cost_params["CSb"], cost_params["COb"], cost_params["Mb"]
    T, B = d.shape
    eps = 1e-4

    # Solve second-stage LP to get optimal solution
    Q_base, u, s, o, i = second_stage_LP_solver(x, d, cost_params)
    alpha = np.zeros_like(d, dtype=np.float64)

    # Compute gradient in reverse order (from last period to first)
    for t in reversed(range(T)):
        for b in range(B):
            # Case 1: Shortage occurs - gradient equals unit shortage cost
            if s[t, b] > eps:
                alpha[t, b] = CSb[b]
                continue

            remaining_inventory = i[t, :, :] - u[t, :, :]

            # Case 2: Last period with remaining inventory in later batches - gradient is 0
            if t == T - 1:
                has_late_batch_remaining = np.any(remaining_inventory[b, 1:] > eps)
                if has_late_batch_remaining:
                    alpha[t, b] = 0.0
                    continue

            # Case 3: Expiration occurs with remaining inventory in earliest batch - gradient equals -COb
            condition_expire = o[t, b] > eps
            condition_earliest_remaining = remaining_inventory[b, 0] > eps
            if condition_expire and condition_earliest_remaining:
                alpha[t, b] = -COb[b]
                continue

            # Case 4: Non-last period with remaining inventory in later batches - gradient = -CHb + next period's gradient
            if t <= T - 2:
                has_late_batch_remaining = np.any(remaining_inventory[b, 1:] > eps)
                if has_late_batch_remaining:
                    alpha[t, b] = -CHb[b] + alpha[t + 1, b]
                    continue

            # Case 5: All inventory depleted - gradient equals unit shortage cost
            all_depleted = np.all(np.abs(remaining_inventory[b, :]) <= eps)
            if all_depleted:
                alpha[t, b] = CSb[b]
                continue

            # Default case (should not occur in optimal solution)
            alpha[t, b] = CSb[b]

    return alpha


def solve_upper_qcqp_multi_period(alpha: np.ndarray, a: np.ndarray, H: np.ndarray, n_solutions: int = 3) -> List[
    np.ndarray]:
    """
    Solve the upper-level QCQP problem to find the worst-case demand direction based on gradient.
    Uses Gurobi's solution pool to return multiple high-quality candidate scenarios.
    """
    T, B = alpha.shape
    if np.all(np.abs(alpha) < 1e-8):
        return [np.tile(np.maximum(a, 0.0), (T, 1)) for _ in range(n_solutions)]

    model = gp.Model("Upper_QCQP_MultiPeriod")
    model = set_gurobi_params(model, is_subproblem=True)

    # Configure solution pool to return multiple top solutions
    model.Params.PoolSolutions = n_solutions * 2
    model.Params.PoolSearchMode = 2  # Search for multiple optimal solutions
    model.Params.PoolGap = 0.1  # Allow 10% optimality gap for pool solutions

    d = model.addVars(T, B, lb=0.0, name="d")

    # Objective: Maximize the linear approximation of Q(X,D)
    model.setObjective(
        gp.quicksum(alpha[t, b] * d[t, b] for t in range(T) for b in range(B)),
        GRB.MAXIMIZE
    )

    # MVCE uncertainty set constraints for each period
    for t in range(T):
        quad_expr = 0.0
        for i in range(B):
            for j in range(B):
                quad_expr += H[i, j] * (d[t, i] - a[i]) * (d[t, j] - a[j])
        model.addConstr(quad_expr <= 1.0, name=f"ellipsoid_constraint_t{t}")

    model.optimize()

    d_opt_list = []
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        n_return = min(model.SolCount, n_solutions)
        for sol_idx in range(n_return):
            model.Params.SolutionNumber = sol_idx
            d_sol = np.zeros((T, B), dtype=np.float64)
            for t in range(T):
                for b in range(B):
                    d_sol[t, b] = d[t, b].X
            d_opt_list.append(d_sol)

    # Fill with default scenarios if not enough solutions are found
    while len(d_opt_list) < n_solutions:
        d_opt_list.append(np.tile(np.maximum(a, 0.0), (T, 1)))

    model.dispose()
    return d_opt_list


def bcd_subproblem_solve(x_current: np.ndarray, T: int, B: int, a: np.ndarray, H: np.ndarray, cost_params: dict,
                         initial_worst_d_list: List[np.ndarray] = None) -> Tuple[
    List[np.ndarray], List[float], List[Tuple]]:
    """
    Solve the I-C&CG subproblem using the multi-start gradient ascent (GA) method.
    Avoids KKT transformation and big-M linearization entirely.
    Args:
        initial_worst_d_list: List of initial worst-case scenarios from previous iterations
    Returns:
        d_worst_list: List of top N worst-case demand scenarios
        Q_worst_list: Corresponding second-stage costs
        solution_detail_list: Detailed second-stage solutions for each scenario
    """
    inner_epsilon = EXPERIMENT_PARAMS["bcd_inner_epsilon"]
    max_inner_iter = EXPERIMENT_PARAMS["bcd_max_inner_iter"]
    n_scenarios = EXPERIMENT_PARAMS["n_multi_cut_scenarios"]
    Mb = cost_params["Mb"]

    n_initial_points = 500  # Number of initial points for multi-start sampling
    global_pool_size = 3  # Keep top 3 highest-cost scenarios in the global pool
    global_pool = []

    def _sanitize_d(d_array):
        """Ensure all demand values are non-negative."""
        return np.maximum(d_array, 0.0)

    def _add_to_pool(Q, d, u, s, o, i):
        """Add a new scenario to the global pool, keeping only the top N highest-cost ones."""
        d_clean = _sanitize_d(d)

        # Check for duplicate scenarios
        is_duplicate = False
        for item in global_pool:
            if np.linalg.norm(d_clean - item[1]) < 1e-3:
                is_duplicate = True
                break
        if is_duplicate:
            return

        global_pool.append((Q, d_clean.copy(), u.copy(), s.copy(), o.copy(), i.copy()))
        # Sort pool in descending order of cost
        global_pool.sort(key=lambda x: -x[0])
        if len(global_pool) > global_pool_size:
            global_pool.pop()

    # Step 1: Generate initial points by sampling the ellipsoid boundary
    initial_d_list = sample_ellipsoid_points(a, H, n_samples=n_initial_points)

    # Add historical worst-case scenarios from the master problem to initial pool
    if initial_worst_d_list is not None:
        for d_init in initial_worst_d_list:
            d_init_full = np.tile(d_init, (T, 1)) if d_init.ndim == 1 else d_init
            d_init_full = _sanitize_d(d_init_full)
            Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_init_full, cost_params)
            _add_to_pool(Q_val, d_init_full, u_val, s_val, o_val, i_val)

    # Evaluate all initial points and populate the global pool
    for d_init in initial_d_list:
        d_init_full = np.tile(d_init, (T, 1))
        d_init_full = _sanitize_d(d_init_full)
        Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_init_full, cost_params)
        _add_to_pool(Q_val, d_init_full, u_val, s_val, o_val, i_val)

    # Fallback: ensure pool is not empty
    if len(global_pool) == 0:
        d_backup_list = sample_ellipsoid_points(a, H, n_samples=1)
        d_backup = np.tile(d_backup_list[0], (T, 1))
        Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_backup, cost_params)
        _add_to_pool(Q_val, d_backup, u_val, s_val, o_val, i_val)

    # Step 2: Main gradient ascent iteration loop
    k = 0
    Q_prev_best = -np.inf

    while k < max_inner_iter:
        k += 1

        # Select the best scenario from the pool as the starting point
        d_k = _sanitize_d(global_pool[0][1].copy())
        Q_k, u_k, s_k, o_k, i_k = second_stage_LP_solver(x_current, d_k, cost_params)
        alpha_k = compute_gradient(x_current, d_k, cost_params)

        _add_to_pool(Q_k, d_k, u_k, s_k, o_k, i_k)

        # Check for convergence of inner loop
        current_best_Q = global_pool[0][0]
        if k > 1:
            gap = np.abs((current_best_Q - Q_prev_best) / (current_best_Q + 1e-8))
            if gap < inner_epsilon:
                break

        Q_prev_best = current_best_Q

        # Solve upper-level QCQP to find next candidate scenarios
        d_next_list = solve_upper_qcqp_multi_period(alpha_k, a, H, n_solutions=n_scenarios)

        # Evaluate new candidates and update global pool
        for d_next in d_next_list:
            d_next_clean = _sanitize_d(d_next)
            Q_next, u_next, s_next, o_next, i_next = second_stage_LP_solver(x_current, d_next_clean, cost_params)
            _add_to_pool(Q_next, d_next_clean, u_next, s_next, o_next, i_next)

    # Fill pool with default scenarios if needed
    while len(global_pool) < n_scenarios:
        d_fill = np.tile(np.maximum(a, 0.0), (T, 1))
        Q_fill, u_fill, s_fill, o_fill, i_fill = second_stage_LP_solver(x_current, d_fill, cost_params)
        _add_to_pool(Q_fill, d_fill, u_fill, s_fill, o_fill, i_fill)

    # Extract top N results
    final_results = global_pool[:n_scenarios]

    d_worst_list = [item[1] for item in final_results]
    Q_worst_list = [item[0] for item in final_results]
    solution_detail_list = [(item[2], item[3], item[4], item[5]) for item in final_results]

    return d_worst_list, Q_worst_list, solution_detail_list


def solve_iccg_master_problem(T: int, B: int, scenarios: List[np.ndarray], cost_params: dict):
    """
    Solve the master problem for the I-C&CG algorithm.
    Also returns the worst-case scenario in the current master problem for subproblem initialization.
    """
    CF, CVb, CHb, CSb, COb, Mb, M_big = cost_params["CF"], cost_params["CVb"], cost_params["CHb"], cost_params["CSb"], \
    cost_params["COb"], cost_params["Mb"], cost_params["M_big"]

    model = gp.Model("I_CCG_MP")
    model = set_gurobi_params(model)

    # First-stage decision variables
    x = model.addVars(T, B, lb=0, name="x")
    z = model.addVars(T, vtype=GRB.BINARY, name="z")
    theta = model.addVar(lb=0, name="theta")

    # Objective: Minimize total first-stage cost + worst-case second-stage cost
    model.setObjective(
        gp.quicksum(CF * z[t] for t in range(T)) +
        gp.quicksum(CVb[b] * x[t, b] for t in range(T) for b in range(B)) +
        theta,
        GRB.MINIMIZE
    )

    # Fixed cost constraint
    for t in range(T):
        for b in range(B):
            model.addConstr(x[t, b] <= M_big * z[t], name=f"fixed_cost_{t}_{b}")

    num_scenarios = len(scenarios)
    i = {}
    u = {}
    s = {}
    o = {}

    for j in range(num_scenarios):
        d_j = scenarios[j]
        i[j] = model.addVars(T + 1, B, Mb, lb=0, name=f"i_{j}")
        u[j] = model.addVars(T, B, Mb, lb=0, name=f"u_{j}")
        s[j] = model.addVars(T, B, lb=0, name=f"s_{j}")
        o[j] = model.addVars(T, B, lb=0, name=f"o_{j}")

        # Cutting plane constraint
        scene_cost_hold = gp.quicksum(CHb[b] * i[j][t, b, m] for t in range(T) for b in range(B) for m in range(Mb))
        scene_cost_stockout = gp.quicksum(CSb[b] * s[j][t, b] for t in range(T) for b in range(B))
        scene_cost_expire = gp.quicksum(COb[b] * o[j][t, b] for t in range(T) for b in range(B))
        model.addConstr(theta >= scene_cost_hold + scene_cost_stockout + scene_cost_expire, name=f"cut_plane_{j}")

        # Initial inventory
        for b in range(B):
            for m in range(Mb - 1):
                model.addConstr(i[j][0, b, m] == 0, name=f"init_i_{j}_{b}_{m}")

        for t in range(T):
            for b_idx in range(B):
                # Replenishment
                model.addConstr(i[j][t, b_idx, Mb - 1] == x[t, b_idx], name=f"replenish_{j}_{t}_{b_idx}")

                # Demand balance
                model.addConstr(
                    gp.quicksum(u[j][t, b_idx, m] for m in range(Mb)) + s[j][t, b_idx] >= d_j[t, b_idx],
                    name=f"demand_balance_{j}_{t}_{b_idx}"
                )
                # Total usage cannot exceed demand
                model.addConstr(gp.quicksum(u[j][t, b_idx, m] for m in range(Mb)) <= d_j[t, b_idx],
                                name=f"sum_u_le_d_{j}_{t}_{b_idx}")

                # Usage cannot exceed available inventory
                for m in range(Mb):
                    model.addConstr(u[j][t, b_idx, m] <= i[j][t, b_idx, m], name=f"inv_upper_{j}_{t}_{b_idx}_{m}")

                # Expiration
                model.addConstr(o[j][t, b_idx] == i[j][t, b_idx, 0] - u[j][t, b_idx, 0], name=f"expire_{j}_{t}_{b_idx}")

                # Inventory carryover
                for m in range(1, Mb):
                    model.addConstr(i[j][t + 1, b_idx, m - 1] == i[j][t, b_idx, m] - u[j][t, b_idx, m],
                                    name=f"shift_{j}_{t}_{b_idx}_{m}")

    model.optimize()

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        x_val = np.array([[x[t, b].X for b in range(B)] for t in range(T)])
        z_val = np.array([1 if z[t].X > 0.5 else 0 for t in range(T)])
        obj_val = model.ObjVal

        # Identify the worst-case scenario in the current master problem
        max_scene_cost = -1
        worst_scene_d = None

        for j in range(num_scenarios):
            current_cost = 0.0
            for t in range(T):
                for b in range(B):
                    for m in range(Mb):
                        current_cost += CHb[b] * i[j][t, b, m].X
                    current_cost += CSb[b] * s[j][t, b].X
                    current_cost += COb[b] * o[j][t, b].X

            if current_cost > max_scene_cost:
                max_scene_cost = current_cost
                worst_scene_d = scenarios[j]

        return model, x_val, z_val, obj_val, worst_scene_d, max_scene_cost
    else:
        return model, None, None, None, None, None


def run_iccg_algorithm(T: int, B: int, a: np.ndarray, H: np.ndarray, cost_params: dict) -> Dict:
    """
    Run the complete I-C&CG algorithm for a given problem instance.
    Returns:
        Dictionary of algorithm performance results
    """
    start_time = time.time()
    epsilon = EXPERIMENT_PARAMS["epsilon"]
    max_outer_iter = EXPERIMENT_PARAMS["max_outer_iter"]
    max_consecutive_duplicate = EXPERIMENT_PARAMS["max_consecutive_duplicate"]
    n_multi_cut = EXPERIMENT_PARAMS["n_multi_cut_scenarios"]
    CF, CVb = cost_params["CF"], cost_params["CVb"]

    # Initialize with the non-negative center of the uncertainty set
    D = [np.maximum(np.tile(a, (T, 1)), 0.0)]
    LB, UB = -np.inf, np.inf
    iter_count = 0
    converged = False
    x_opt = None
    consecutive_duplicate = 0

    while iter_count < max_outer_iter:
        iter_count += 1

        # Solve master problem
        model_mp, x_current, z_current, MP_obj, d_worst_in_history, theta_actual = solve_iccg_master_problem(T, B, D,
                                                                                                             cost_params)

        if x_current is None:
            break

        LB = max(LB, MP_obj)  # Update lower bound

        # Solve subproblem using multi-start gradient ascent
        d_worst_list, Q_worst_list, solution_detail_list = bcd_subproblem_solve(
            x_current, T, B, a, H, cost_params,
            initial_worst_d_list=[d_worst_in_history]
        )

        # Calculate current upper bound
        cost_1st_fixed = np.sum(CF * z_current)
        cost_1st_variable = np.sum(CVb * x_current)
        max_Q_worst = max(Q_worst_list)

        candidate_UB = cost_1st_fixed + cost_1st_variable + max_Q_worst
        UB = candidate_UB
        x_opt = x_current.copy()

        # Check for convergence
        current_gap = np.abs((UB - LB) / LB) if LB > 1e-4 else np.inf
        if current_gap <= epsilon and LB > 1e-6:
            converged = True
            break

        # Add new non-duplicate scenarios to the master problem
        new_scene_count = 0
        for d_worst in d_worst_list:
            d_worst_safe = np.maximum(d_worst, 0.0)

            is_new = not any(np.all(np.abs(d_worst_safe - d_exist) < 1e-3) for d_exist in D)
            if is_new:
                D.append(d_worst_safe)
                new_scene_count += 1

        # Terminate if no new scenarios are found for multiple consecutive iterations
        if new_scene_count == 0:
            consecutive_duplicate += 1
            if consecutive_duplicate >= max_consecutive_duplicate:
                break
        else:
            consecutive_duplicate = 0

    total_time = time.time() - start_time

    return {
        "algorithm": "I_CCG",
        "T": T, "B": B,
        "converged": converged,
        "total_time": total_time,
        "iter_count": iter_count,
        "LB": LB, "UB": UB,
        "TC_opt": UB if converged else np.nan,
        "x_opt": x_opt,
        "scenario_count": len(D),
        "time_per_iter": total_time / iter_count if iter_count > 0 else np.nan
    }


# ==============================================================================
# ===================== Part 4: Main Experiment Runner =====================
# ==============================================================================

def single_experiment_run(B: int, T: int, repeat: int):
    """
    Run a single experiment instance for both algorithms with identical demand data and uncertainty set.
    Args:
        repeat: Replication index (0-based)
    Returns:
        List of results for both algorithms
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment Started: B={B}, T={T}, Replication={repeat + 1}/{EXPERIMENT_PARAMS['n_repeat']}")
    print(f"{'=' * 60}")

    cost_params = get_cost_params(B)

    # Generate identical historical demand data and uncertainty set for both algorithms
    demand_history = generate_truncated_mvn(B, EXPERIMENT_PARAMS["n_history"], EXPERIMENT_PARAMS["rho"])
    a, H = khachiyan_algorithm(demand_history)

    results = []

    # Run Classical C&CG algorithm
    print(f"\n>>> Running Classical C&CG algorithm...")
    try:
        classic_result = run_classic_ccg_algorithm(T, B, a, H, cost_params)
        results.append(classic_result)
        print(
            f"Classical C&CG Completed: Time={classic_result['total_time']:.2f}s, Iterations={classic_result['iter_count']}, Optimal Cost={classic_result['TC_opt']:.2f}")
    except Exception as e:
        print(f"Classical C&CG Failed: {str(e)}")
        results.append({
            "algorithm": "Classic_CCG", "T": T, "B": B, "converged": False,
            "total_time": np.nan, "iter_count": np.nan, "LB": np.nan, "UB": np.nan,
            "TC_opt": np.nan, "x_opt": None, "scenario_count": np.nan, "time_per_iter": np.nan
        })

    # Run I-C&CG algorithm
    print(f"\n>>> Running I-C&CG algorithm...")
    try:
        iccg_result = run_iccg_algorithm(T, B, a, H, cost_params)
        results.append(iccg_result)
        print(
            f"I-C&CG Completed: Time={iccg_result['total_time']:.2f}s, Iterations={iccg_result['iter_count']}, Optimal Cost={iccg_result['TC_opt']:.2f}")
    except Exception as e:
        print(f"I-C&CG Failed: {str(e)}")
        results.append({
            "algorithm": "I_CCG", "T": T, "B": B, "converged": False,
            "total_time": np.nan, "iter_count": np.nan, "LB": np.nan, "UB": np.nan,
            "TC_opt": np.nan, "x_opt": None, "scenario_count": np.nan, "time_per_iter": np.nan
        })

    return results


def main():
    """Main function to run all algorithm comparison experiments."""
    os.makedirs(EXPERIMENT_PARAMS["output_dir"], exist_ok=True)
    print("=" * 80)
    print("Algorithm Comparison Experiment (Section 5.1 of the Manuscript)")
    print("Classical C&CG vs Improved C&CG (I-C&CG)")
    print("=" * 80)
    print(f"Experimental Parameters:")
    print(f"  - Number of blood product categories B: {EXPERIMENT_PARAMS['B_list']}")
    print(f"  - Planning horizon length T: {EXPERIMENT_PARAMS['T_list']}")
    print(f"  - Number of independent replications: {EXPERIMENT_PARAMS['n_repeat']}")
    print(f"  - Time limit per solver run: {EXPERIMENT_PARAMS['time_limit']} seconds")
    print(f"  - Convergence tolerance: {EXPERIMENT_PARAMS['epsilon']}")
    print("=" * 80)

    # Generate all experiment tasks
    tasks = [(B, T, r) for B in EXPERIMENT_PARAMS["B_list"]
             for T in EXPERIMENT_PARAMS["T_list"]
             for r in range(EXPERIMENT_PARAMS["n_repeat"])]

    # Run all experiments sequentially
    all_results = []
    for B, T, r in tasks:
        run_results = single_experiment_run(B, T, r)
        all_results.extend(run_results)

    # Process results
    df = pd.DataFrame(all_results)

    # Calculate summary statistics grouped by B, T, and algorithm
    summary_df = df.groupby(['B', 'T', 'algorithm']).agg({
        'converged': 'mean',
        'total_time': 'mean',
        'iter_count': 'mean',
        'TC_opt': 'mean',
        'time_per_iter': 'mean',
        'scenario_count': 'mean'
    }).reset_index()

    # Generate Table 2 in the paper format
    print("\n" + "=" * 120)
    print("Table 2: Computational Performance Comparison between Classical C&CG and I-C&CG Algorithms")
    print("=" * 120)
    print(
        f"{'B':<3} {'T':<3} {'Algorithm':<12} {'Avg. Iterations':<15} {'Avg. Total Time (s)':<20} {'Avg. Time per Iter (s)':<20} {'Avg. Optimal Cost ($)':<20} {'Convergence Rate (%)':<15}")
    print("-" * 120)

    for B in EXPERIMENT_PARAMS["B_list"]:
        for T in EXPERIMENT_PARAMS["T_list"]:
            # Classical C&CG results
            classic_row = summary_df[
                (summary_df['B'] == B) & (summary_df['T'] == T) & (summary_df['algorithm'] == 'Classic_CCG')]
            if not classic_row.empty:
                cr = classic_row.iloc[0]
                converged_rate = cr['converged'] * 100
                time_str = f"{cr['total_time']:.2f}" if not np.isnan(cr['total_time']) else ">3600"
                iter_str = f"{cr['iter_count']:.1f}" if not np.isnan(cr['iter_count']) else "-"
                cost_str = f"{cr['TC_opt']:.2f}" if not np.isnan(cr['TC_opt']) else "-"
                per_iter_str = f"{cr['time_per_iter']:.2f}" if not np.isnan(cr['time_per_iter']) else "-"
                print(
                    f"{B:<3} {T:<3} {'Classic_CCG':<12} {iter_str:<15} {time_str:<20} {per_iter_str:<20} {cost_str:<20} {converged_rate:<15.1f}")

            # I-C&CG results
            iccg_row = summary_df[
                (summary_df['B'] == B) & (summary_df['T'] == T) & (summary_df['algorithm'] == 'I_CCG')]
            if not iccg_row.empty:
                ir = iccg_row.iloc[0]
                converged_rate = ir['converged'] * 100
                time_str = f"{ir['total_time']:.2f}" if not np.isnan(ir['total_time']) else ">3600"
                iter_str = f"{ir['iter_count']:.1f}" if not np.isnan(ir['iter_count']) else "-"
                cost_str = f"{ir['TC_opt']:.2f}" if not np.isnan(ir['TC_opt']) else "-"
                per_iter_str = f"{ir['time_per_iter']:.2f}" if not np.isnan(ir['time_per_iter']) else "-"
                print(
                    f"{B:<3} {T:<3} {'I_CCG':<12} {iter_str:<15} {time_str:<20} {per_iter_str:<20} {cost_str:<20} {converged_rate:<15.1f}")

            print("-" * 120)

    # Save results to CSV files
    try:
        raw_output_path = os.path.join(EXPERIMENT_PARAMS["output_dir"], "algorithm_comparison_raw.csv")
        # Convert numpy arrays to strings for CSV storage
        df['x_opt_str'] = df['x_opt'].apply(lambda x: np.array2string(x, separator=',') if x is not None else "")
        df.drop('x_opt', axis=1).to_csv(raw_output_path, index=False, encoding="utf-8-sig")

        summary_output_path = os.path.join(EXPERIMENT_PARAMS["output_dir"], "algorithm_comparison_summary.csv")
        summary_df.to_csv(summary_output_path, index=False, encoding="utf-8-sig")

        print(f"\nResults saved successfully:")
        print(f"  - Raw experimental data: {raw_output_path}")
        print(f"  - Summary statistics: {summary_output_path}")
    except PermissionError:
        print(
            "\nWarning: Permission denied when writing to output directory. Saving to current working directory instead...")
        df['x_opt_str'] = df['x_opt'].apply(lambda x: np.array2string(x, separator=',') if x is not None else "")
        df.drop('x_opt', axis=1).to_csv("algorithm_comparison_raw_backup.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv("algorithm_comparison_summary_backup.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()