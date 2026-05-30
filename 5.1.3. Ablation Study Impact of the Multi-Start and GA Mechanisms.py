# ablation_study.py
# Ablation Study: Comparing Full I-C&CG with Two Variants (No Multi-Start / No Gradient Ascent)
# Improved C&CG Algorithm Based on Original Multi-Cut BCD Framework

import os
import numpy as np
from scipy.stats import multivariate_normal
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd
from typing import Tuple, List, Dict, Optional
from joblib import Parallel, delayed

# ==============================================================================
# Global Parameters (Consistent with Original Code, Adjustable as Needed)
# ==============================================================================
np.random.seed(20260401)

BASE_COSTS = {
    "CVb": [200, 200, 200, 200, 200],
    "CHb": [10, 10, 10, 10, 10],
    "CSb": [2000, 2000, 2000, 2000, 2000],
    "COb": [200, 200, 200, 200, 200]
}

EXPERIMENT_PARAMS = {
    "CF": 200, "epsilon": 1e-5, "max_outer_iter": 200, "time_limit": 3600,
    "M_big": 1e4,
    "B_list": [3],  # Adjust as needed: [2,3,5]
    "T_list": [3],  # Adjust as needed: [3,5,7]
    "n_history": 365,
    "n_repeat": 100,  # Number of repetitions per configuration
    "rho": 0.3,
    "output_dir": "./ablation_results",
    "n_jobs": 1,
    "gurobi_threads": 1,
    "gurobi_outputflag": 0,
    "Mb": 5,
    "demand_upper_bound_ratio": 5.0,
    "max_consecutive_duplicate": 3,
    "bcd_inner_epsilon": 1e-4,
    "bcd_max_inner_iter": 100,
    "grad_epsilon": 1e-4,
    "n_multi_cut_scenarios": 1,
    "debug_infeasibility": False,  # Disable debug output for ablation experiments
}


def get_cost_params(B: int, base_costs: dict) -> dict:
    """Extract and format cost parameters for a given number of blood product types"""
    return {
        "CF": EXPERIMENT_PARAMS["CF"],
        "CVb": np.array(base_costs["CVb"][:B]),
        "CHb": np.array(base_costs["CHb"][:B]),
        "CSb": np.array(base_costs["CSb"][:B]),
        "COb": np.array(base_costs["COb"][:B]),
        "Mb": EXPERIMENT_PARAMS["Mb"],
        "M_big": EXPERIMENT_PARAMS["M_big"]
    }


def set_gurobi_params(model: gp.Model, is_subproblem: bool = False, force_output: bool = False):
    """Configure Gurobi solver parameters for optimal performance and numerical stability"""
    model.Params.Threads = EXPERIMENT_PARAMS["gurobi_threads"]
    model.Params.OutputFlag = 1 if force_output else EXPERIMENT_PARAMS["gurobi_outputflag"]
    model.Params.TimeLimit = EXPERIMENT_PARAMS["time_limit"]
    model.Params.Method = -1
    model.Params.MIPFocus = 1
    model.Params.Cuts = 1
    model.Params.DualReductions = 0
    model.Params.PreDual = 0
    model.Params.PreQLinearize = 1
    model.Params.Presolve = 1
    model.Params.FeasibilityTol = 1e-4
    model.Params.OptimalityTol = 1e-4
    model.Params.IntFeasTol = 1e-4
    model.Params.InfUnbdInfo = 1
    model.Params.NumericFocus = 1
    if is_subproblem:
        model.Params.NumericFocus = 1
        model.Params.NonConvex = 2
        model.Params.MIPGap = 1e-4
    return model


# ==============================================================================
# Data Generation and MVCE Parameter Calculation (Identical to Original Code)
# ==============================================================================
def khachiyan_algorithm(data: np.ndarray, tol: float = 1e-4, max_iter: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Khachiyan's algorithm to compute the Minimum Volume Covering Ellipsoid (MVCE)
    for a set of points in R^n. Returns the ellipsoid center and shape matrix.
    """
    n_samples, n_dim = data.shape
    Q = np.vstack([data.T, np.ones(n_samples)])
    p = np.ones(n_samples) / n_samples
    reg = 1e-4 * np.eye(Q.shape[0])

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

        omega = (g_max - n_dim - 1) / ((n_dim + 1) * (g_max - 1)) if g_max > n_dim + 1 else 0
        p = (1 - omega) * p
        p[j] += omega
        p /= np.sum(p)

    a = data.T @ p
    centered_data = data - a
    Sigma = centered_data.T @ np.diag(p) @ centered_data
    H = np.linalg.inv(n_dim * Sigma + 1e-4 * np.eye(n_dim))
    return a, H


def generate_truncated_mvn(B: int, n_samples: int, rho: float = 0.5) -> np.ndarray:
    """Generate non-negative truncated multivariate normal demand data with specified correlation"""
    if B == 2:
        mu = np.array([40, 30])
    elif B == 3:
        mu = np.array([40, 30, 25])
    elif B == 5:
        mu = np.array([40, 35, 30, 25, 20])
    else:
        raise ValueError(f"Unsupported number of blood products: B={B}")

    sigma = 0.5 * mu
    Sigma = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            Sigma[i, j] = sigma[i] ** 2 if i == j else rho * sigma[i] * sigma[j]
    Sigma += 1e-4 * np.eye(B)

    mvn = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)
    return np.maximum(mvn.rvs(size=n_samples), 0)


def sample_ellipsoid_points(
        a: np.ndarray,
        H: np.ndarray,
        n_samples: int
) -> List[np.ndarray]:
    """Generate uniformly distributed points on the boundary of the MVCE uncertainty set"""
    B = len(a)
    samples = []

    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        reg = 1e-4 * np.eye(B)
        L = np.linalg.cholesky(H + reg)

    if B == 2:
        # Polar coordinate uniform sampling for 2D
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        for theta in angles:
            z = np.array([np.cos(theta), np.sin(theta)])
            y = np.linalg.solve(L.T, z)
            d_sample = a + y
            d_sample = np.maximum(d_sample, 0.0)
            samples.append(d_sample)
    else:
        # Fibonacci sphere sampling for 3D, random normalized sampling for higher dimensions
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
        for i in range(n_samples):
            if B == 3:
                y_coord = 1 - (i / float(n_samples - 1)) * 2
                radius = np.sqrt(1 - y_coord * y_coord)
                theta = phi * i
                x_coord = np.cos(theta) * radius
                z_coord = np.sin(theta) * radius
                z = np.array([x_coord, y_coord, z_coord])
            else:
                z = np.random.randn(B)
                norm_z = np.linalg.norm(z)
                if norm_z > 1e-8:
                    z /= norm_z

            y = np.linalg.solve(L.T, z)
            d_sample = a + y
            d_sample = np.maximum(d_sample, 0.0)
            samples.append(d_sample)

    return samples


# ==============================================================================
# Second-Stage LP Solver (Identical to Original Code)
# ==============================================================================
def second_stage_LP_solver(
        x: np.ndarray,
        d: np.ndarray,
        cost_params: dict
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the second-stage linear programming problem for given first-stage
    procurement decisions and realized demand scenario.
    Returns optimal cost and corresponding decision variables.
    """
    CHb = cost_params["CHb"]
    CSb = cost_params["CSb"]
    COb = cost_params["COb"]
    Mb = cost_params["Mb"]
    T, B = x.shape

    model = gp.Model("SecondStage_LP")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.NumericFocus = 3
    model.Params.FeasibilityTol = 1e-4
    model.Params.OptimalityTol = 1e-4

    # Decision variables
    i = model.addVars(T + 1, B, Mb, lb=0.0, name="i")  # Inventory by remaining shelf life
    u = model.addVars(T, B, Mb, lb=0.0, name="u")  # Usage quantity by shelf life
    s = model.addVars(T, B, lb=0.0, name="s")  # Shortage quantity
    o = model.addVars(T, B, lb=0.0, name="o")  # Expired quantity

    # Objective: Minimize total second-stage cost (holding + shortage + expiration)
    total_cost = 0.0
    for t in range(T):
        for b in range(B):
            for m in range(Mb):
                total_cost += CHb[b] * i[t, b, m]
        for b in range(B):
            total_cost += CSb[b] * s[t, b]
            total_cost += COb[b] * o[t, b]
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Initial inventory constraints: no carryover before planning horizon
    for b in range(B):
        for m in range(Mb - 1):
            model.addConstr(i[0, b, m] == 0.0, name=f"init_i_{b}_{m}")

    # Core inventory dynamics constraints
    for t in range(T):
        for b_idx in range(B):
            # Newly procured blood has full shelf life
            model.addConstr(i[t, b_idx, Mb - 1] == x[t, b_idx], name=f"replenish_{t}_{b_idx}")

            # Demand balance: total usage + shortage ≥ demand
            model.addConstr(
                gp.quicksum(u[t, b_idx, m] for m in range(Mb)) + s[t, b_idx] >= d[t, b_idx],
                name=f"demand_balance_{t}_{b_idx}"
            )

            # Total usage cannot exceed demand (no over-issuance)
            model.addConstr(
                gp.quicksum(u[t, b_idx, m] for m in range(Mb)) <= d[t, b_idx],
                name=f"sum_u_le_d_{t}_{b_idx}"
            )

            # Usage cannot exceed available inventory for each shelf life batch
            for m in range(Mb):
                model.addConstr(u[t, b_idx, m] <= i[t, b_idx, m], name=f"inv_upper_{t}_{b_idx}_{m}")

            # Expiration: unused inventory with 1 period remaining life expires
            model.addConstr(o[t, b_idx] == i[t, b_idx, 0] - u[t, b_idx, 0], name=f"expire_{t}_{b_idx}")

            # Inventory carryover: remaining shelf life decreases by 1 period
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
        raise Exception(f"Second-stage LP failed with status code: {model.status}")


def compute_gradient(
        x: np.ndarray,
        d: np.ndarray,
        cost_params: dict
) -> np.ndarray:
    """
    Compute the gradient of the second-stage cost function Q(X,D) with respect to demand D
    using a reverse-order loop and priority-based inventory status classification.
    """
    CHb = cost_params["CHb"]
    CSb = cost_params["CSb"]
    COb = cost_params["COb"]
    Mb = cost_params["Mb"]
    T, B = d.shape
    eps = 1e-4

    Q_base, u, s, o, i = second_stage_LP_solver(x, d, cost_params)
    alpha = np.zeros_like(d, dtype=np.float64)

    # Reverse-order calculation from last period to first
    for t in reversed(range(T)):
        for b in range(B):
            # Case 1: Shortage occurs - marginal cost equals unit shortage cost
            if s[t, b] > eps:
                alpha[t, b] = CSb[b]
                continue

            remaining_inventory = i[t, :, :] - u[t, :, :]

            # Case 2: Last period with excess inventory - no future cost
            if t == T - 1:
                if np.any(remaining_inventory[b, 1:] > eps):
                    alpha[t, b] = 0.0
                    continue

            # Case 3: Excess inventory in the oldest batch - marginal cost equals expiration cost
            if o[t, b] > eps and remaining_inventory[b, 0] > eps:
                alpha[t, b] = -COb[b]
                continue

            # Case 4: Excess inventory in younger batches - carryover to next period
            if t <= T - 2:
                if np.any(remaining_inventory[b, 1:] > eps):
                    alpha[t, b] = -CHb[b] + alpha[t + 1, b]
                    continue

            # Case 5: No remaining inventory - marginal cost equals unit shortage cost
            if np.all(np.abs(remaining_inventory[b, :]) <= eps):
                alpha[t, b] = CSb[b]
                continue

            # Default case: treat as shortage for robustness
            alpha[t, b] = CSb[b]

    return alpha


def solve_upper_qcqp_multi_period(
        alpha: np.ndarray,
        a: np.ndarray,
        H: np.ndarray,
        n_solutions: int = EXPERIMENT_PARAMS["n_multi_cut_scenarios"]
) -> List[np.ndarray]:
    """
    Solve the upper-level quadratically constrained quadratic programming (QCQP) problem
    to find the worst-case demand direction for gradient ascent.
    Returns multiple candidate solutions for multi-cut generation.
    """
    T, B = alpha.shape
    if np.all(np.abs(alpha) < 1e-8):
        return [np.tile(np.maximum(a, 0.0), (T, 1)) for _ in range(n_solutions)]

    model = gp.Model("Upper_QCQP_MultiPeriod")
    model = set_gurobi_params(model, is_subproblem=True)
    model.Params.PoolSolutions = n_solutions * 2
    model.Params.PoolSearchMode = 2
    model.Params.PoolGap = 0.1

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

    # Fill with ellipsoid center if not enough solutions
    while len(d_opt_list) < n_solutions:
        d_opt_list.append(np.tile(np.maximum(a, 0.0), (T, 1)))

    model.dispose()
    return d_opt_list


# ==============================================================================
# Ablation Study: Three Subproblem Solver Variants
# ==============================================================================
def subproblem_full(
        x_current: np.ndarray,
        T: int,
        B: int,
        a: np.ndarray,
        H: np.ndarray,
        cost_params: dict,
        initial_worst_d_list: List[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[Tuple]]:
    """Full I-C&CG subproblem (multi-start initialization + gradient ascent iterations)"""
    inner_epsilon = EXPERIMENT_PARAMS["bcd_inner_epsilon"]
    max_inner_iter = EXPERIMENT_PARAMS["bcd_max_inner_iter"]
    Mb = cost_params["Mb"]
    n_initial_points = 500
    global_pool_size = 3
    global_pool = []

    def _sanitize_d(d_array):
        """Ensure non-negative demand values"""
        return np.maximum(d_array, 0.0)

    def _add_to_pool(Q, d, u, s, o, i):
        """Add a scenario to the global pool, maintaining uniqueness and size limit"""
        d_clean = _sanitize_d(d)
        # Check for duplicate scenarios
        for item in global_pool:
            if np.linalg.norm(d_clean - item[1]) < 1e-3:
                return
        global_pool.append((Q, d_clean.copy(), u.copy(), s.copy(), o.copy(), i.copy()))
        # Keep only the top N highest-cost scenarios
        global_pool.sort(key=lambda x: -x[0])
        if len(global_pool) > global_pool_size:
            global_pool.pop()

    # Multi-start initialization sampling
    initial_d_list = sample_ellipsoid_points(a, H, n_samples=n_initial_points)

    # Add historical worst-case scenarios from master problem
    if initial_worst_d_list is not None:
        for d_init in initial_worst_d_list:
            d_init_full = np.tile(d_init, (T, 1)) if d_init.ndim == 1 else d_init
            d_init_full = _sanitize_d(d_init_full)
            Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_init_full, cost_params)
            _add_to_pool(Q_val, d_init_full, u_val, s_val, o_val, i_val)

    # Evaluate all initial samples
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

    # Gradient ascent iterations
    k = 0
    Q_prev_best = -np.inf
    while k < max_inner_iter:
        k += 1
        # Start from the best scenario in the pool
        if len(global_pool) == 0:
            d_k = np.tile(np.maximum(a, 0.0), (T, 1))
        else:
            d_k = _sanitize_d(global_pool[0][1].copy())

        # Evaluate current scenario and compute gradient
        Q_k, u_k, s_k, o_k, i_k = second_stage_LP_solver(x_current, d_k, cost_params)
        alpha_k = compute_gradient(x_current, d_k, cost_params)
        _add_to_pool(Q_k, d_k, u_k, s_k, o_k, i_k)

        # Check convergence
        current_best_Q = global_pool[0][0]
        if k > 1:
            gap = np.abs((current_best_Q - Q_prev_best) / (current_best_Q + 1e-8))
            if gap < inner_epsilon:
                break
        Q_prev_best = current_best_Q

        # Find next candidate scenario via QCQP
        d_next_list = solve_upper_qcqp_multi_period(alpha_k, a, H,
                                                    n_solutions=EXPERIMENT_PARAMS["n_multi_cut_scenarios"])
        for d_next in d_next_list:
            d_next_clean = _sanitize_d(d_next)
            Q_next, u_next, s_next, o_next, i_next = second_stage_LP_solver(x_current, d_next_clean, cost_params)
            _add_to_pool(Q_next, d_next_clean, u_next, s_next, o_next, i_next)

    # Ensure we have enough scenarios for multi-cut generation
    while len(global_pool) < EXPERIMENT_PARAMS["n_multi_cut_scenarios"]:
        d_fill = np.tile(np.maximum(a, 0.0), (T, 1))
        Q_fill, u_fill, s_fill, o_fill, i_fill = second_stage_LP_solver(x_current, d_fill, cost_params)
        _add_to_pool(Q_fill, d_fill, u_fill, s_fill, o_fill, i_fill)

    # Return top N worst-case scenarios
    final_results = global_pool[:EXPERIMENT_PARAMS["n_multi_cut_scenarios"]]
    d_worst_list = [item[1] for item in final_results]
    Q_worst_list = [item[0] for item in final_results]
    solution_detail_list = [(item[2], item[3], item[4], item[5]) for item in final_results]

    return d_worst_list, Q_worst_list, solution_detail_list


def subproblem_no_multistart(
        x_current: np.ndarray,
        T: int,
        B: int,
        a: np.ndarray,
        H: np.ndarray,
        cost_params: dict,
        initial_worst_d_list: List[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[Tuple]]:
    """
    No multi-start variant: Only uses the worst-case scenario from the master problem
    as the single initial point, then performs gradient ascent iterations.
    """
    inner_epsilon = EXPERIMENT_PARAMS["bcd_inner_epsilon"]
    max_inner_iter = EXPERIMENT_PARAMS["bcd_max_inner_iter"]
    global_pool_size = 3
    global_pool = []

    def _sanitize_d(d_array):
        return np.maximum(d_array, 0.0)

    def _add_to_pool(Q, d, u, s, o, i):
        d_clean = _sanitize_d(d)
        for item in global_pool:
            if np.linalg.norm(d_clean - item[1]) < 1e-3:
                return
        global_pool.append((Q, d_clean.copy(), u.copy(), s.copy(), o.copy(), i.copy()))
        global_pool.sort(key=lambda x: -x[0])
        if len(global_pool) > global_pool_size:
            global_pool.pop()

    # Single initial point: from historical worst-case scenario, or ellipsoid center if none
    if initial_worst_d_list is not None and len(initial_worst_d_list) > 0:
        d_start = initial_worst_d_list[0]
        d_start_full = np.tile(d_start, (T, 1)) if d_start.ndim == 1 else d_start
    else:
        d_start_full = np.tile(np.maximum(a, 0.0), (T, 1))

    d_start_full = _sanitize_d(d_start_full)
    Q_start, u_start, s_start, o_start, i_start = second_stage_LP_solver(x_current, d_start_full, cost_params)
    _add_to_pool(Q_start, d_start_full, u_start, s_start, o_start, i_start)

    # Gradient ascent iterations (same as full version)
    k = 0
    Q_prev_best = -np.inf
    while k < max_inner_iter:
        k += 1
        d_k = _sanitize_d(global_pool[0][1].copy())
        Q_k, u_k, s_k, o_k, i_k = second_stage_LP_solver(x_current, d_k, cost_params)
        alpha_k = compute_gradient(x_current, d_k, cost_params)
        _add_to_pool(Q_k, d_k, u_k, s_k, o_k, i_k)

        current_best_Q = global_pool[0][0]
        if k > 1:
            gap = np.abs((current_best_Q - Q_prev_best) / (current_best_Q + 1e-8))
            if gap < inner_epsilon:
                break
        Q_prev_best = current_best_Q

        d_next_list = solve_upper_qcqp_multi_period(alpha_k, a, H,
                                                    n_solutions=EXPERIMENT_PARAMS["n_multi_cut_scenarios"])
        for d_next in d_next_list:
            d_next_clean = _sanitize_d(d_next)
            Q_next, u_next, s_next, o_next, i_next = second_stage_LP_solver(x_current, d_next_clean, cost_params)
            _add_to_pool(Q_next, d_next_clean, u_next, s_next, o_next, i_next)

    while len(global_pool) < EXPERIMENT_PARAMS["n_multi_cut_scenarios"]:
        d_fill = np.tile(np.maximum(a, 0.0), (T, 1))
        Q_fill, u_fill, s_fill, o_fill, i_fill = second_stage_LP_solver(x_current, d_fill, cost_params)
        _add_to_pool(Q_fill, d_fill, u_fill, s_fill, o_fill, i_fill)

    final_results = global_pool[:EXPERIMENT_PARAMS["n_multi_cut_scenarios"]]
    d_worst_list = [item[1] for item in final_results]
    Q_worst_list = [item[0] for item in final_results]
    solution_detail_list = [(item[2], item[3], item[4], item[5]) for item in final_results]

    return d_worst_list, Q_worst_list, solution_detail_list


def subproblem_no_gradient(
        x_current: np.ndarray,
        T: int,
        B: int,
        a: np.ndarray,
        H: np.ndarray,
        cost_params: dict,
        initial_worst_d_list: List[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[Tuple]]:
    """
    No gradient ascent variant: Only evaluates multi-start samples,
    no iterative gradient ascent (does not call solve_upper_qcqp_multi_period).
    """
    n_initial_points = 500
    global_pool_size = 3
    global_pool = []

    def _sanitize_d(d_array):
        return np.maximum(d_array, 0.0)

    def _add_to_pool(Q, d, u, s, o, i):
        d_clean = _sanitize_d(d)
        for item in global_pool:
            if np.linalg.norm(d_clean - item[1]) < 1e-3:
                return
        global_pool.append((Q, d_clean.copy(), u.copy(), s.copy(), o.copy(), i.copy()))
        global_pool.sort(key=lambda x: -x[0])
        if len(global_pool) > global_pool_size:
            global_pool.pop()

    # Multi-start initialization sampling (same as full version)
    initial_d_list = sample_ellipsoid_points(a, H, n_samples=n_initial_points)

    if initial_worst_d_list is not None:
        for d_init in initial_worst_d_list:
            d_init_full = np.tile(d_init, (T, 1)) if d_init.ndim == 1 else d_init
            d_init_full = _sanitize_d(d_init_full)
            Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_init_full, cost_params)
            _add_to_pool(Q_val, d_init_full, u_val, s_val, o_val, i_val)

    # Evaluate all initial samples
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

    # No gradient ascent loop - directly return the best scenarios from the pool
    final_results = global_pool[:EXPERIMENT_PARAMS["n_multi_cut_scenarios"]]
    d_worst_list = [item[1] for item in final_results]
    Q_worst_list = [item[0] for item in final_results]
    solution_detail_list = [(item[2], item[3], item[4], item[5]) for item in final_results]

    return d_worst_list, Q_worst_list, solution_detail_list


# ==============================================================================
# Unified C&CG Execution Framework (Supports Specified Subproblem Variants)
# ==============================================================================
def run_ccg_with_subproblem_mode(
        T: int,
        B: int,
        a: np.ndarray,
        H: np.ndarray,
        cost_params: dict,
        mode: str  # "full", "no_multistart", "no_gradient"
) -> Dict:
    """Run the C&CG algorithm with a specified subproblem solver mode"""
    start_time = time.time()
    epsilon = EXPERIMENT_PARAMS["epsilon"]
    max_outer_iter = EXPERIMENT_PARAMS["max_outer_iter"]
    max_consecutive_duplicate = EXPERIMENT_PARAMS["max_consecutive_duplicate"]
    n_multi_cut = EXPERIMENT_PARAMS["n_multi_cut_scenarios"]
    CF, CVb = cost_params["CF"], cost_params["CVb"]

    # Initialize with ellipsoid center scenario
    D = [np.maximum(np.tile(a, (T, 1)), 0.0)]
    LB, UB = -np.inf, np.inf
    iter_count = 0
    x_opt = None
    consecutive_duplicate = 0

    # Select subproblem function based on specified mode
    if mode == "full":
        subproblem_func = subproblem_full
    elif mode == "no_multistart":
        subproblem_func = subproblem_no_multistart
    elif mode == "no_gradient":
        subproblem_func = subproblem_no_gradient
    else:
        raise ValueError("Mode must be 'full', 'no_multistart' or 'no_gradient'")

    while iter_count < max_outer_iter:
        iter_count += 1

        # Solve the master problem
        model_mp, x_current, z_current, MP_obj, d_worst_in_history, theta_actual = solve_master_problem(T, B, D,
                                                                                                        cost_params)
        if x_current is None:
            break
        LB = max(LB, MP_obj)

        # Solve the subproblem to find worst-case scenarios
        d_worst_list, Q_worst_list, _ = subproblem_func(
            x_current, T, B, a, H, cost_params,
            initial_worst_d_list=[d_worst_in_history] if d_worst_in_history is not None else None
        )

        # Calculate current upper bound
        cost_1st_fixed = np.sum(CF * z_current)
        cost_1st_variable = np.sum(CVb * x_current)
        max_Q_worst = max(Q_worst_list)
        candidate_UB = cost_1st_fixed + cost_1st_variable + max_Q_worst
        UB = candidate_UB
        x_opt = x_current.copy()

        # Check convergence
        current_gap = np.abs((UB - LB) / LB) if LB > 1e-4 else np.inf
        if current_gap <= epsilon and LB > 1e-6:
            break

        # Add new unique scenarios to the master problem
        new_scene_count = 0
        for d_worst in d_worst_list:
            d_worst_safe = np.maximum(d_worst, 0.0)
            is_new = not any(np.all(np.abs(d_worst_safe - d_exist) < 1e-3) for d_exist in D)
            if is_new:
                D.append(d_worst_safe)
                new_scene_count += 1

        # Terminate if no new scenarios are found for consecutive iterations
        if new_scene_count == 0:
            consecutive_duplicate += 1
            if consecutive_duplicate >= max_consecutive_duplicate:
                break
        else:
            consecutive_duplicate = 0

    # Return comprehensive results
    return {
        "mode": mode,
        "T": T,
        "B": B,
        "converged": (current_gap <= epsilon),
        "total_time": time.time() - start_time,
        "iter_count": iter_count,
        "LB": LB,
        "UB": UB,
        "TC_opt": UB,
        "scenario_count": len(D),
    }


def solve_master_problem(T: int, B: int, scenarios: List[np.ndarray], cost_params: dict):
    """
    Solve the master problem of the C&CG algorithm with the current set of worst-case scenarios.
    Returns the model, optimal first-stage decisions, objective value, and worst-case scenario.
    """
    CF, CVb, CHb, CSb, COb, Mb, M_big = [cost_params[k] for k in ["CF", "CVb", "CHb", "CSb", "COb", "Mb", "M_big"]]

    model = gp.Model("Classic_CCG_MP")
    model = set_gurobi_params(model, force_output=False)

    # First-stage decision variables
    x = model.addVars(T, B, lb=0, name="x")  # Order quantity
    z = model.addVars(T, vtype=GRB.BINARY, name="z")  # Order indicator
    theta = model.addVar(lb=0, name="theta")  # Auxiliary variable for worst-case cost

    # Objective: Minimize first-stage cost + worst-case second-stage cost
    model.setObjective(
        gp.quicksum(CF * z[t] for t in range(T)) +
        gp.quicksum(CVb[b] * x[t, b] for t in range(T) for b in range(B)) +
        theta,
        GRB.MINIMIZE
    )

    # Fixed cost constraint: order only if z[t] = 1
    for t in range(T):
        for b in range(B):
            model.addConstr(x[t, b] <= M_big * z[t], name=f"fixed_cost_{t}_{b}")

    # Second-stage variables and constraints for each scenario
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

        # Cut plane constraint: theta ≥ second-stage cost for each scenario
        scene_cost_hold = gp.quicksum(CHb[b] * i[j][t, b, m] for t in range(T) for b in range(B) for m in range(Mb))
        scene_cost_stockout = gp.quicksum(CSb[b] * s[j][t, b] for t in range(T) for b in range(B))
        scene_cost_expire = gp.quicksum(COb[b] * o[j][t, b] for t in range(T) for b in range(B))
        model.addConstr(theta >= scene_cost_hold + scene_cost_stockout + scene_cost_expire, name=f"cut_plane_{j}")

        # Initial inventory constraints
        for b in range(B):
            for m in range(Mb - 1):
                model.addConstr(i[j][0, b, m] == 0, name=f"init_i_{j}_{b}_{m}")

        # Core inventory dynamics constraints (same as second-stage LP)
        for t in range(T):
            for b_idx in range(B):
                model.addConstr(i[j][t, b_idx, Mb - 1] == x[t, b_idx], name=f"replenish_{j}_{t}_{b_idx}")
                model.addConstr(
                    gp.quicksum(u[j][t, b_idx, m] for m in range(Mb)) + s[j][t, b_idx] >= d_j[t, b_idx],
                    name=f"demand_balance_{j}_{t}_{b_idx}"
                )
                for m in range(Mb):
                    model.addConstr(u[j][t, b_idx, m] <= i[j][t, b_idx, m], name=f"inv_upper_{j}_{t}_{b_idx}_{m}")
                model.addConstr(o[j][t, b_idx] == i[j][t, b_idx, 0] - u[j][t, b_idx, 0], name=f"expire_{j}_{t}_{b_idx}")
                for m in range(1, Mb):
                    model.addConstr(i[j][t + 1, b_idx, m - 1] == i[j][t, b_idx, m] - u[j][t, b_idx, m],
                                    name=f"shift_{j}_{t}_{b_idx}_{m}")
                model.addConstr(gp.quicksum(u[j][t, b_idx, m] for m in range(Mb)) <= d_j[t, b_idx],
                                name=f"sum_u_le_d_{j}_{t}_{b_idx}")

    model.optimize()
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        x_val = np.array([[x[t, b].X for b in range(B)] for t in range(T)])
        z_val = np.array([1 if z[t].X > 0.5 else 0 for t in range(T)])
        obj_val = model.ObjVal
        theta_val = theta.X

        # Identify the worst-case scenario (used as initial point for subproblem)
        max_scene_cost = -1
        worst_scene_idx = -1
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
                worst_scene_idx = j
                worst_scene_d = scenarios[j]

        theta_actual = max_scene_cost
        return model, x_val, z_val, obj_val, worst_scene_d, theta_actual
    else:
        return model, None, None, None, None, None


# ==============================================================================
# Main Function for Ablation Study
# ==============================================================================
def single_ablation_run(B: int, T: int, mode: str, repeat: int):
    """Execute a single ablation experiment run for a given configuration"""
    print(f"  Running {mode}  B={B} T={T} repeat={repeat + 1}")
    cost_params = get_cost_params(B, BASE_COSTS)
    demand_history = generate_truncated_mvn(B, EXPERIMENT_PARAMS["n_history"], EXPERIMENT_PARAMS["rho"])
    a, H = khachiyan_algorithm(demand_history)
    result = run_ccg_with_subproblem_mode(T, B, a, H, cost_params, mode=mode)
    result["repeat"] = repeat
    return result


def main():
    """Main entry point for the ablation study: run all configurations and aggregate results"""
    os.makedirs(EXPERIMENT_PARAMS["output_dir"], exist_ok=True)
    modes = ["full", "no_multistart", "no_gradient"]
    all_results = []

    for B in EXPERIMENT_PARAMS["B_list"]:
        for T in EXPERIMENT_PARAMS["T_list"]:
            for mode in modes:
                print(f"\nStarting ablation experiment: B={B}, T={T}, mode={mode}")
                for r in range(EXPERIMENT_PARAMS["n_repeat"]):
                    res = single_ablation_run(B, T, mode, r)
                    all_results.append(res)

    # Aggregate and print results
    df = pd.DataFrame(all_results)
    pivot = df.groupby(["B", "T", "mode"]).agg({
        "TC_opt": ["mean", "std"],
        "total_time": ["mean", "std"],
        "iter_count": "mean",
        "scenario_count": "mean"
    }).round(2)

    print("\n" + "=" * 80)
    print("Ablation Study Results Summary")
    print("=" * 80)
    print(pivot.to_string())

    # Save detailed results to CSV
    csv_path = os.path.join(EXPERIMENT_PARAMS["output_dir"], "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to {csv_path}")


if __name__ == "__main__":
    main()