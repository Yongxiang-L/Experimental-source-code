import os
import numpy as np
from scipy.stats import multivariate_normal
import gurobipy as gp
from gurobipy import GRB
import time
import pandas as pd
from typing import Tuple, List, Dict

# ==============================================================================
# ===================== Global Parameters and Utility Functions =====================
# ==============================================================================
np.random.seed(20260509)  # Fix random seed for reproducibility

# Table 1 in Section 5.1.1: Baseline cost parameters (uniform for all products)
BASE_COSTS = {
    "CVb": [200, 200, 200, 200, 200],
    "CHb": [10, 10, 10, 10, 10],
    "CSb": [2000, 2000, 2000, 2000, 2000],
    "COb": [200, 200, 200, 200, 200]
}

EXPERIMENT_PARAMS = {
    "CF": 200, "epsilon": 1e-4, "max_outer_iter": 200, "time_limit": 3600,
    "M_big": 1e5,
    "B": 3, "T": 7, "n_history": 365,
    "rho": 0.3, "sigma": 0.5, "output_dir": "./5.4_Experiment_Results",
    "gurobi_threads": 1, "gurobi_outputflag": 0,
    "demand_upper_bound_ratio": 5.0,
    "max_consecutive_duplicate": 3,
    # BCD subproblem parameters
    "bcd_inner_epsilon": 1e-4,
    "bcd_max_inner_iter": 100,
    "grad_epsilon": 1e-6,
    # Multi-cut batch update parameters
    "n_multi_cut_scenarios": 3,
    # Debug parameters
    "debug_infeasibility": True,
    # Section 5.4 specific parameters
    "Mb_list": [3, 4, 5, 6, 7, 8, 9, 10],  # Shelf life gradient specified in the paper
    "n_monte_carlo": 10000,
    "n_repeat_54": 100  # 100 repeated experiments per shelf life level
}


def get_cost_params(B: int, Mb: int, base_costs: dict) -> dict:
    """Retrieve cost parameters for given number of products and shelf life"""
    return {
        "CF": EXPERIMENT_PARAMS["CF"],
        "CVb": np.array(BASE_COSTS["CVb"][:B]),
        "CHb": np.array(BASE_COSTS["CHb"][:B]),
        "CSb": np.array(BASE_COSTS["CSb"][:B]),
        "COb": np.array(BASE_COSTS["COb"][:B]),
        "Mb": Mb,
        "M_big": EXPERIMENT_PARAMS["M_big"]
    }


def set_gurobi_params(model: gp.Model, is_subproblem: bool = False, force_output: bool = False):
    """Configure Gurobi solver parameters for master and subproblems"""
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
    model.Params.NumericFocus = 3

    if is_subproblem:
        model.Params.NumericFocus = 3
        model.Params.NonConvex = 2
        model.Params.MIPGap = 1e-6
    else:
        model.Params.NumericFocus = 3
    return model


# ==============================================================================
# ===================== Infeasibility Diagnosis Tool =====================
# ==============================================================================
def diagnose_infeasibility(model: gp.Model, scenarios: List[np.ndarray], T: int, B: int, Mb: int):
    """Deep diagnosis of master problem infeasibility"""
    print("\n" + "=" * 80)
    print("🔴 Starting deep diagnosis of master problem infeasibility")
    print("=" * 80)

    print("\n1. Distinguish between Infeasible and Unbounded...")
    feas_model = model.copy()
    feas_model.setObjective(0)
    feas_model.Params.OutputFlag = 1
    feas_model.optimize()

    if feas_model.status == GRB.OPTIMAL:
        print("   ✅ Model is feasible! Original problem is Unbounded")
        return
    elif feas_model.status == GRB.INFEASIBLE:
        print("   ❌ Model is confirmed Infeasible")
    else:
        print(f"   ⚠️  Feasibility check failed with status code: {feas_model.status}")
        return

    print("\n2. Computing Irreducible Inconsistent Subsystem (IIS)...")
    try:
        feas_model.computeIIS()
        print(f"   ✅ IIS computed successfully, found {feas_model.NumIISConstrs} conflicting constraints")
    except Exception as e:
        print(f"   ❌ IIS computation failed: {e}")
        return

    print("\n3. Detailed analysis of conflicting constraints:")
    print("-" * 80)
    print(f"{'Constraint Name':<40} {'Constraint Type':<15} {'Conflict Cause'}")
    print("-" * 80)

    conflict_types = {}
    scenario_conflicts = {}

    for constr in feas_model.getConstrs():
        if constr.IISConstr:
            constr_name = constr.ConstrName
            parts = constr_name.split('_')
            constr_type = parts[0] if len(parts) > 0 else "unknown"
            conflict_types[constr_type] = conflict_types.get(constr_type, 0) + 1

            if len(parts) >= 2 and parts[1].isdigit():
                scene_idx = int(parts[1])
                scenario_conflicts[scene_idx] = scenario_conflicts.get(scene_idx, 0) + 1
                if scene_idx < len(scenarios):
                    d_scene = scenarios[scene_idx]
                    max_d = np.max(d_scene)
                    min_d = np.min(d_scene)
                    print(f"{constr_name:<40} {constr_type:<15} Scenario {scene_idx} Range=[{min_d:.2f}, {max_d:.2f}]")
                else:
                    print(f"{constr_name:<40} {constr_type:<15}")
            else:
                print(f"{constr_name:<40} {constr_type:<15}")

    print("\n" + "-" * 80)
    print("4. Root cause analysis summary:")
    print("-" * 80)
    print(f"\n   Conflict constraint type statistics: {conflict_types}")

    if scenario_conflicts:
        print(f"   Conflict scenario statistics: {scenario_conflicts}")
        worst_scene_idx = max(scenario_conflicts, key=scenario_conflicts.get)
        worst_d = scenarios[worst_scene_idx]
        print(f"\n   🎯 Most problematic scenario: Scenario {worst_scene_idx}")
        print(f"      Demand matrix for this scenario:\n{worst_d}")
        print(f"      Maximum demand in this scenario: {np.max(worst_d):.2f}")
        print(f"      Minimum demand in this scenario: {np.min(worst_d):.2f}")

    if 'expire' in conflict_types:
        print("\n   🚨 Expiration quantity constraint conflict")
        print("      Recommendation: Modify expiration constraint to o >= i_0 - u_0")

    print("\n" + "=" * 80)
    print("Diagnosis completed")
    print("=" * 80 + "\n")


# ==============================================================================
# ===================== MVCE Uncertainty Set Calibration Tool =====================
# ==============================================================================
def khachiyan_algorithm(data: np.ndarray, tol: float = 1e-6, max_iter: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 1 in the paper: Khachiyan algorithm for Minimum Volume Covering Ellipsoid (MVCE) calculation
    Implemented strictly according to the paper's formulas with convergence tolerance 1e-6
    """
    n_samples, n_dim = data.shape
    Q = np.vstack([data.T, np.ones(n_samples)])
    p = np.ones(n_samples) / n_samples
    reg = 1e-6 * np.eye(Q.shape[0])

    for _ in range(max_iter):
        Lambda = Q @ np.diag(p) @ Q.T + reg
        try:
            Lambda_inv = np.linalg.inv(Lambda)
        except np.linalg.LinAlgError:
            Lambda += 1e-4 * np.eye(Lambda.shape[0])
            Lambda_inv = np.linalg.inv(Lambda)

        g = np.diag(Q.T @ Lambda_inv @ Q)
        j, g_max = np.argmax(g), g.max()

        if g_max <= 1 + tol:
            break

        omega = (g_max - n_dim - 1) / ((n_dim + 1) * (g_max - 1)) if g_max > n_dim + 1 else 0
        p = (1 - omega) * p
        p[j] += omega
        p /= np.sum(p)

    a = data.T @ p
    centered_data = data - a
    Sigma = centered_data.T @ np.diag(p) @ centered_data
    H = np.linalg.inv(n_dim * Sigma + 1e-6 * np.eye(n_dim))

    return a, H


# ==============================================================================
# ===================== Demand Generation Function (Fixed ρ=0.3, σ=0.5) =====================
# ==============================================================================
def generate_truncated_mvn(
        B: int,
        n_samples: int,
        rho: float = EXPERIMENT_PARAMS["rho"],
        sigma_ratio: float = EXPERIMENT_PARAMS["sigma"]
) -> np.ndarray:
    """
    Section 5.1.1: Generate truncated multivariate normal distribution demand data
    Strictly follows paper parameters: mean [40,35,30,25,20], standard deviation = 0.5 * mean
    """
    full_mu = np.array([40, 35, 30, 25, 20])
    mu = full_mu[:B]
    sigma = sigma_ratio * mu  # Standard deviation = sigma_ratio * mean

    # Construct covariance matrix (strictly according to paper formula)
    Sigma = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            Sigma[i, j] = sigma[i] ** 2 if i == j else rho * sigma[i] * sigma[j]
    Sigma += 1e-6 * np.eye(B)  # Ensure positive definiteness

    mvn = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)
    return np.maximum(mvn.rvs(size=n_samples), 0)  # Non-negative truncation


# ==============================================================================
# ===================== Ellipsoid Boundary Uniform Sampling Tool =====================
# ==============================================================================
def sample_ellipsoid_points(
        a: np.ndarray,
        H: np.ndarray,
        n_samples: int
) -> List[np.ndarray]:
    """
    Section 4.2.2: Generate deterministically uniform points on the ellipsoid boundary
    Uses Fibonacci sphere sampling for B=3 to ensure uniform coverage
    """
    B = len(a)
    samples = []

    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        reg = 1e-6 * np.eye(B)
        L = np.linalg.cholesky(H + reg)

    if B == 3:
        # For B=3: Fibonacci sphere sampling (method specified in the paper)
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
        # General case: Low-discrepancy sequence sampling
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


# ==============================================================================
# ===================== Second-Stage LP Solver (Dynamic Mb Support) =====================
# ==============================================================================
def second_stage_LP_solver(
        x: np.ndarray,
        d: np.ndarray,
        cost_params: dict
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given procurement plan x and demand d, build LP model to solve for minimum second-stage cost
    Supports dynamic adjustment of shelf life Mb
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
    model.Params.FeasibilityTol = 1e-6
    model.Params.OptimalityTol = 1e-6

    # Variable definitions (dynamically adapt to Mb)
    i = model.addVars(T + 1, B, Mb, lb=0.0, name="i")
    u = model.addVars(T, B, Mb, lb=0.0, name="u")
    s = model.addVars(T, B, lb=0.0, name="s")
    o = model.addVars(T, B, lb=0.0, name="o")

    # Objective: Minimize holding + stockout + expiration costs
    total_cost = 0.0
    for t in range(T):
        for b in range(B):
            for m in range(Mb):
                total_cost += CHb[b] * i[t, b, m]
            total_cost += CSb[b] * s[t, b]
            total_cost += COb[b] * o[t, b]
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Constraint definitions (strictly follow formulas 7-15 in the paper)
    # 1. Initial inventory constraints
    for b in range(B):
        for m in range(Mb - 1):
            model.addConstr(i[0, b, m] == 0.0, name=f"init_i_{b}_{m}")

    # 2. Periodic inventory dynamics constraints
    for t in range(T):
        for b_idx in range(B):
            # Current period procurement receipt
            model.addConstr(i[t, b_idx, Mb - 1] == x[t, b_idx], name=f"replenish_{t}_{b_idx}")

            # Demand balance constraint
            model.addConstr(
                gp.quicksum(u[t, b_idx, m] for m in range(Mb)) + s[t, b_idx] >= d[t, b_idx],
                name=f"demand_balance_{t}_{b_idx}"
            )
            model.addConstr(
                gp.quicksum(u[t, b_idx, m] for m in range(Mb)) <= d[t, b_idx],
                name=f"sum_u_le_d_{t}_{b_idx}"
            )

            # Inventory consumption upper bound constraint
            for m in range(Mb):
                model.addConstr(u[t, b_idx, m] <= i[t, b_idx, m], name=f"inv_upper_{t}_{b_idx}_{m}")

            # Expiration quantity constraint
            model.addConstr(o[t, b_idx] == i[t, b_idx, 0] - u[t, b_idx, 0], name=f"expire_{t}_{b_idx}")

            # Inventory carryover constraint
            for m in range(1, Mb):
                model.addConstr(
                    i[t + 1, b_idx, m - 1] == i[t, b_idx, m] - u[t, b_idx, m],
                    name=f"shift_{t}_{b_idx}_{m}"
                )

    # Solve the model
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
        raise Exception(f"Second-stage LP solution failed with status code: {model.status}")


# ==============================================================================
# ===================== Analytical Gradient Calculation (Dynamic Mb Support) =====================
# ==============================================================================
def compute_gradient(
        x: np.ndarray,
        d: np.ndarray,
        cost_params: dict
) -> np.ndarray:
    """
    Formula 55 in the paper: Analytical gradient calculation based on reverse-order loop
    Implemented strictly according to the paper's priority rules
    """
    Q_base, u, s, o, i = second_stage_LP_solver(x, d, cost_params)
    CHb = cost_params["CHb"]
    CSb = cost_params["CSb"]
    COb = cost_params["COb"]
    Mb = cost_params["Mb"]
    T, B = d.shape
    eps = 1e-6

    alpha = np.zeros_like(d, dtype=np.float64)

    for t in reversed(range(T)):
        for b in range(B):
            # Rule 1: If stockout exists, gradient equals stockout cost
            if s[t, b] > eps:
                alpha[t, b] = CSb[b]
                continue

            remaining_inventory = i[t, b, :] - u[t, b, :]

            # Rule 2: Last period with remaining newer batches, gradient = 0
            if t == T - 1 and np.any(remaining_inventory[1:] > eps):
                alpha[t, b] = 0.0
                continue

            # Rule 3: Expiration occurs with remaining oldest batch, gradient = -expiration cost
            if o[t, b] > eps and remaining_inventory[0] > eps:
                alpha[t, b] = -COb[b]
                continue

            # Rule 4: Non-last period with remaining newer batches, gradient = -holding cost + next period gradient
            if t < T - 1 and np.any(remaining_inventory[1:] > eps):
                alpha[t, b] = -CHb[b] + alpha[t + 1, b]
                continue

            # Rule 5: All batches exhausted, gradient equals stockout cost
            alpha[t, b] = CSb[b]

    return alpha


# ==============================================================================
# ===================== Upper Bound QCQP Problem Solver =====================
# ==============================================================================
def solve_upper_qcqp_multi_period(
        alpha: np.ndarray,
        a: np.ndarray,
        H: np.ndarray,
        n_solutions: int = EXPERIMENT_PARAMS["n_multi_cut_scenarios"]
) -> List[np.ndarray]:
    """
    Formulas 57-59 in the paper: Solve upper bound QCQP problem for MVCE uncertainty set
    Supports batch return of multiple optimal solutions for multi-cut plane updates
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

    model.setObjective(
        gp.quicksum(alpha[t, b] * d[t, b] for t in range(T) for b in range(B)),
        GRB.MAXIMIZE
    )

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

    while len(d_opt_list) < n_solutions:
        d_opt_list.append(np.tile(np.maximum(a, 0.0), (T, 1)))

    model.dispose()
    return d_opt_list


# ==============================================================================
# ===================== BCD Subproblem Solver (Core of I-C&CG Algorithm) =====================
# ==============================================================================
def bcd_subproblem_solve(
        x_current: np.ndarray,
        T: int,
        B: int,
        uncertainty_params: dict,
        cost_params: dict,
        initial_worst_d_list: List[np.ndarray] = None
) -> Tuple[List[np.ndarray], List[float], List[Tuple]]:
    """
    Section 4.2: Gradient ascent-based BCD subproblem solver
    Includes multi-start initialization, global optimal pool, and batch multi-cut plane updates
    """
    inner_epsilon = EXPERIMENT_PARAMS["bcd_inner_epsilon"]
    max_inner_iter = EXPERIMENT_PARAMS["bcd_max_inner_iter"]
    Mb = cost_params["Mb"]
    a = uncertainty_params["a"]
    H = uncertainty_params["H"]

    n_initial_points = 500
    global_pool_size = 3
    global_pool = []

    def _sanitize_d(d_array):
        return np.maximum(d_array, 0.0)

    def _add_to_pool(Q, d, u, s, o, i):
        d_clean = _sanitize_d(d)

        is_duplicate = False
        for item in global_pool:
            if np.linalg.norm(d_clean - item[1]) < 1e-3:
                is_duplicate = True
                break
        if is_duplicate:
            return

        global_pool.append((Q, d_clean.copy(), u.copy(), s.copy(), o.copy(), i.copy()))
        global_pool.sort(key=lambda x: -x[0])
        if len(global_pool) > global_pool_size:
            global_pool.pop()

    print(f"    [BCD] Phase 1: Initialization (total {n_initial_points} points)...")

    # Generate initial points on ellipsoid boundary
    initial_d_list = sample_ellipsoid_points(a, H, n_samples=n_initial_points)

    if initial_worst_d_list is not None:
        print(f"    [BCD] Injecting {len(initial_worst_d_list)} master problem worst scenarios as initial points...")
        for d_init in initial_worst_d_list:
            d_init_full = np.tile(d_init, (T, 1)) if d_init.ndim == 1 else d_init
            d_init_full = _sanitize_d(d_init_full)
            Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_init_full, cost_params)
            _add_to_pool(Q_val, d_init_full, u_val, s_val, o_val, i_val)

    for d_init in initial_d_list:
        d_init_full = np.tile(d_init, (T, 1))
        d_init_full = _sanitize_d(d_init_full)
        Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_init_full, cost_params)
        _add_to_pool(Q_val, d_init_full, u_val, s_val, o_val, i_val)

    if len(global_pool) > 0:
        print(f"    [BCD] Initialization complete, current worst cost: {global_pool[0][0]:.2f}")
    else:
        print(f"    [BCD] Warning: Pool is empty, using mean as fallback...")
        d_backup = np.tile(np.maximum(a, 0.0), (T, 1))
        Q_val, u_val, s_val, o_val, i_val = second_stage_LP_solver(x_current, d_backup, cost_params)
        _add_to_pool(Q_val, d_backup, u_val, s_val, o_val, i_val)

    k = 0
    Q_prev_best = -np.inf

    print(f"    [BCD] Phase 2: Hybrid gradient ascent iteration...")

    while k < max_inner_iter:
        k += 1

        if len(global_pool) == 0:
            d_k = np.tile(np.maximum(a, 0.0), (T, 1))
        else:
            d_k = _sanitize_d(global_pool[0][1].copy())

        Q_k, u_k, s_k, o_k, i_k = second_stage_LP_solver(x_current, d_k, cost_params)
        alpha_k = compute_gradient(x_current, d_k, cost_params)

        _add_to_pool(Q_k, d_k, u_k, s_k, o_k, i_k)

        current_best_Q = global_pool[0][0]
        if k > 1:
            gap = np.abs((current_best_Q - Q_prev_best) / (current_best_Q + 1e-8))
            if gap < inner_epsilon:
                print(f"    [BCD] Converged at iteration {k}, Gap={gap:.6f}")
                break

        Q_prev_best = current_best_Q

        # Batch solve multiple upper bound solutions for multi-cut plane updates
        d_next_list = solve_upper_qcqp_multi_period(alpha_k, a, H)

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

    print(f"    [BCD] Completed, returning Top-{len(d_worst_list)} scenarios, worst cost={Q_worst_list[0]:.2f}")

    return d_worst_list, Q_worst_list, solution_detail_list


# ==============================================================================
# ===================== Master Problem Solver (Dynamic Mb Support) =====================
# ==============================================================================
def solve_master_problem(T: int, B: int, Mb: int, scenarios: List[np.ndarray], cost_params: dict):
    """Build and solve the I-C&CG master problem with dynamic shelf life Mb support"""
    CF, CVb, CHb, CSb, COb, M_big = [cost_params[k] for k in ["CF", "CVb", "CHb", "CSb", "COb", "M_big"]]

    print(f"  Building master problem, number of scenarios: {len(scenarios)}, shelf life: {Mb} days")

    model = gp.Model("I-C&CG_MP")
    model = set_gurobi_params(model, force_output=(len(scenarios) > 10))

    x = model.addVars(T, B, lb=0, name="x")
    z = model.addVars(T, vtype=GRB.BINARY, name="z")
    theta = model.addVar(lb=0, name="theta")

    model.setObjective(
        gp.quicksum(CF * z[t] for t in range(T)) +
        gp.quicksum(CVb[b] * x[t, b] for t in range(T) for b in range(B)) +
        theta,
        GRB.MINIMIZE
    )

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

        scene_cost_hold = gp.quicksum(CHb[b] * i[j][t, b, m] for t in range(T) for b in range(B) for m in range(Mb))
        scene_cost_stockout = gp.quicksum(CSb[b] * s[j][t, b] for t in range(T) for b in range(B))
        scene_cost_expire = gp.quicksum(COb[b] * o[j][t, b] for t in range(T) for b in range(B))
        model.addConstr(theta >= scene_cost_hold + scene_cost_stockout + scene_cost_expire, name=f"cut_plane_{j}")

        for b in range(B):
            for m in range(Mb - 1):
                model.addConstr(i[j][0, b, m] == 0, name=f"init_i_{j}_{b}_{m}")

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
                                name=f"sum1_u_le_d_{j}_{t}_{b_idx}")

    model.optimize()

    if model.status == GRB.INF_OR_UNBD and EXPERIMENT_PARAMS["debug_infeasibility"]:
        diagnose_infeasibility(model, scenarios, T, B, Mb)
        return model, None, None, None, None, None

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        x_val = np.array([[x[t, b].X for b in range(B)] for t in range(T)])
        z_val = np.array([1 if z[t].X > 0.5 else 0 for t in range(T)])
        obj_val = model.ObjVal
        theta_val = theta.X

        fixed_purchase_cost = sum(CF * z_val[t] for t in range(T))
        variable_purchase_cost = sum(CVb[b] * x_val[t, b] for t in range(T) for b in range(B))
        first_stage_total = fixed_purchase_cost + variable_purchase_cost

        # Identify worst-case scenario
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

        print(f"\n  ============== [Master Problem Debug] ==============")
        print(f"    First-stage total cost: {first_stage_total:.2f}")
        print(f"    Master problem theta: {theta_val:.2f}")
        print(f"    Internally computed worst scenario cost: {theta_actual:.2f} (Scenario {worst_scene_idx})")
        print(f"  ===================================================\n")

        return model, x_val, z_val, obj_val, worst_scene_d, theta_actual
    else:
        print(f"  Master problem solution failed with status code: {model.status}")
        return model, None, None, None, None, None


# ==============================================================================
# ===================== I-C&CG Main Algorithm =====================
# ==============================================================================
def run_classic_ccg_algorithm(
        T: int,
        B: int,
        Mb: int,
        uncertainty_params: dict,
        cost_params: dict
) -> Dict:
    """Run the Improved Column-and-Constraint Generation (I-C&CG) algorithm to solve the two-stage robust optimization model"""
    start_time = time.time()
    epsilon = EXPERIMENT_PARAMS["epsilon"]
    max_outer_iter = EXPERIMENT_PARAMS["max_outer_iter"]
    max_consecutive_duplicate = EXPERIMENT_PARAMS["max_consecutive_duplicate"]
    n_multi_cut = EXPERIMENT_PARAMS["n_multi_cut_scenarios"]
    CF, CVb = cost_params["CF"], cost_params["CVb"]
    a = uncertainty_params["a"]

    # Initialize scenario set
    initial_d = np.maximum(np.tile(a, (T, 1)), 0.0)
    D = [initial_d]
    LB, UB = -np.inf, np.inf
    iter_count = 0
    converged = False
    x_opt = None
    consecutive_duplicate = 0

    print(f"Starting I-C&CG solution (B={B}, T={T}, Mb={Mb} days, batch scenarios per iteration={n_multi_cut})...")

    while iter_count < max_outer_iter:
        iter_count += 1

        print(f"  Iteration {iter_count}: Solving master problem...")
        model_mp, x_current, z_current, MP_obj, d_worst_in_history, theta_actual = solve_master_problem(T, B, Mb, D,
                                                                                                        cost_params)

        if x_current is None:
            print("  ❌ Master problem solution failed, terminating algorithm")
            break

        LB = max(LB, MP_obj)

        print(f"  Iteration {iter_count}: Solving BCD subproblem...")

        d_worst_list, Q_worst_list, solution_detail_list = bcd_subproblem_solve(
            x_current, T, B, uncertainty_params, cost_params,
            initial_worst_d_list=[d_worst_in_history]
        )

        cost_1st_fixed = np.sum(CF * z_current)
        cost_1st_variable = np.sum(CVb * x_current)
        max_Q_worst = max(Q_worst_list)

        candidate_UB = cost_1st_fixed + cost_1st_variable + max_Q_worst
        UB = candidate_UB
        x_opt = x_current.copy()

        current_gap = np.abs((UB - LB) / LB) if LB > 1e-6 else np.inf

        print(f"  Iteration {iter_count:2d} | LB={LB:10.2f} | UB={UB:10.2f} | Gap={current_gap:.6f}")
        print(f"    (Master worst={theta_actual:.2f}, Subproblem worst={max_Q_worst:.2f}, Using={max_Q_worst:.2f})")

        if current_gap <= epsilon and LB > 1e-6:
            converged = True
            print(f"  ✅ I-C&CG converged successfully!")
            break

        new_scene_count = 0
        for d_worst in d_worst_list:
            d_worst_safe = np.maximum(d_worst, 0.0)

            is_new = not any(np.all(np.abs(d_worst_safe - d_exist) < 1e-3) for d_exist in D)
            if is_new:
                D.append(d_worst_safe)
                new_scene_count += 1

        if new_scene_count == 0:
            consecutive_duplicate += 1
            print(f"  No new scenarios this iteration ({consecutive_duplicate}/{max_consecutive_duplicate})\n")
            if consecutive_duplicate >= max_consecutive_duplicate:
                break
        else:
            consecutive_duplicate = 0
            print(f"  Added {new_scene_count} new scenarios this iteration\n")

    return {
        "algorithm": "MVCE_I-C&CG", "T": T, "B": B, "Mb": Mb, "converged": converged,
        "total_time": time.time() - start_time, "iter_count": iter_count,
        "LB": LB, "UB": UB, "TC_opt": UB, "x_opt": x_opt, "scenario_count": len(D),
        "n_multi_cut": n_multi_cut
    }


# ==============================================================================
# ===================== Monte Carlo Simulation Evaluation Function =====================
# ==============================================================================
def monte_carlo_evaluation(
        x_opt: np.ndarray,
        B: int,
        T: int,
        Mb: int,
        rho: float,
        sigma_ratio: float,
        cost_params: dict,
        n_samples: int = EXPERIMENT_PARAMS["n_monte_carlo"],
        test_demand: np.ndarray = None
) -> Dict:
    """
    Section 5.2.2: Monte Carlo simulation for out-of-sample performance evaluation
    Outputs all 7 core metrics required for Table 5 in the paper
    """
    CF = cost_params["CF"]
    CVb = cost_params["CVb"]

    # Calculate first-stage cost (fixed)
    z_opt = np.any(x_opt > 1e-6, axis=1).astype(int)
    cost_1st_fixed = np.sum(CF * z_opt)
    cost_1st_variable = np.sum(CVb * x_opt)
    avg_first_stage = cost_1st_fixed + cost_1st_variable

    # Use provided fixed test set or generate new one
    if test_demand is None:
        demand_samples = generate_truncated_mvn(B, n_samples * T, rho, sigma_ratio)
        demand_samples = demand_samples.reshape(n_samples, T, B)
    else:
        demand_samples = test_demand
        n_samples = demand_samples.shape[0]

    total_holding = 0.0
    total_stockout = 0.0
    total_expiration = 0.0
    total_stockout_units = 0.0
    total_demand_units = 0.0

    for i in range(n_samples):
        d = demand_samples[i]
        Q, u, s, o, i_inv = second_stage_LP_solver(x_opt, d, cost_params)

        # Calculate individual cost components
        holding = np.sum(cost_params["CHb"] * np.sum(i_inv[:T], axis=2))
        total_holding += holding

        stockout_cost = np.sum(cost_params["CSb"] * s)
        total_stockout += stockout_cost
        total_stockout_units += np.sum(s)

        expiration_cost = np.sum(cost_params["COb"] * o)
        total_expiration += expiration_cost

        total_demand_units += np.sum(d)

    # Calculate averages
    avg_holding = total_holding / n_samples
    avg_stockout = total_stockout / n_samples
    avg_expiration = total_expiration / n_samples
    avg_second_stage = avg_holding + avg_stockout + avg_expiration
    avg_total = avg_first_stage + avg_second_stage
    avg_stockout_rate = (total_stockout_units / total_demand_units) * 100 if total_demand_units > 0 else 0.0

    return {
        "avg_first_stage": avg_first_stage,
        "avg_holding": avg_holding,
        "avg_stockout": avg_stockout,
        "avg_expiration": avg_expiration,
        "avg_second_stage": avg_second_stage,
        "avg_total": avg_total,
        "avg_stockout_rate": avg_stockout_rate
    }


# ==============================================================================
# ===================== Section 5.4: Main Function for Shelf Life Impact Experiment =====================
# Strictly follows experimental design in Section 5.4.1
# Parameters: B=3, T=7, all other parameters consistent with the paper
# ==============================================================================
def run_experiment_54():
    """
    Run complete experiment for Section 5.4: Impact of shelf life on MVCE model performance
    Strictly follows paper's experimental design:
    - B=3 blood products, T=7-day planning horizon
    - Shelf life gradient: 3, 4, 5, 6, 7 days
    - Fixed demand correlation ρ=0.3, volatility σ=0.5
    - 10 independent repeats per shelf life level
    - 10,000 Monte Carlo simulations per experiment
    """
    # Core experiment parameters (strictly set according to Section 5.4.1)
    B = EXPERIMENT_PARAMS["B"]  # 3 blood product categories
    T = EXPERIMENT_PARAMS["T"]  # 7-day planning horizon
    rho = EXPERIMENT_PARAMS["rho"]  # Fixed demand correlation coefficient 0.3
    sigma = EXPERIMENT_PARAMS["sigma"]  # Fixed demand volatility coefficient 0.5
    Mb_list = EXPERIMENT_PARAMS["Mb_list"]  # [3,4,5,6,7] days
    n_repeat = EXPERIMENT_PARAMS["n_repeat_54"]  # 10 repeats per shelf life level
    n_history = EXPERIMENT_PARAMS["n_history"]  # 365 days of historical data
    n_monte_carlo = EXPERIMENT_PARAMS["n_monte_carlo"]  # 10,000 Monte Carlo simulations
    output_dir = EXPERIMENT_PARAMS["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for Mb in Mb_list:
        print(f"\n{'=' * 80}")
        print(f"Starting experiment: Maximum shelf life Mb={Mb} days")
        print(f"{'=' * 80}")

        # Get cost parameters for current shelf life
        cost_params = get_cost_params(B, Mb, BASE_COSTS)

        for repeat in range(n_repeat):
            print(f"\n--- Repeat experiment {repeat + 1}/{n_repeat} ---")

            # 1. Generate historical demand data (strictly follows truncated multivariate normal distribution)
            print(f"Generating historical demand data (n={n_history} days)...")
            demand_history = generate_truncated_mvn(B, n_history, rho, sigma)

            # 2. Calibrate MVCE uncertainty set (Khachiyan algorithm, 100% coverage of historical samples)
            print("Calibrating MVCE uncertainty set...")
            a, H = khachiyan_algorithm(demand_history)
            mvce_params = {"a": a, "H": H}

            # 3. Solve MVCE model using I-C&CG algorithm
            print("\nSolving MVCE model...")
            mvce_result = run_classic_ccg_algorithm(T, B, Mb, mvce_params, cost_params)
            if mvce_result["x_opt"] is None:
                print("MVCE model solution failed, skipping this repeat")
                continue

            # 4. Generate fixed Monte Carlo test set (ensure fair evaluation)
            print(f"\nGenerating fixed Monte Carlo test set (n={n_monte_carlo})...")
            test_demand = generate_truncated_mvn(B, n_monte_carlo * T, rho, sigma)
            test_demand = test_demand.reshape(n_monte_carlo, T, B)

            # 5. Monte Carlo out-of-sample evaluation
            print("Evaluating MVCE model via Monte Carlo simulation...")
            mvce_mc = monte_carlo_evaluation(
                mvce_result["x_opt"], B, T, Mb, rho, sigma, cost_params, n_monte_carlo, test_demand
            )

            # 6. Save complete results (includes all metrics for Table 5)
            result_entry = {
                "Mb": Mb,
                "repeat": repeat + 1,
                "avg_first_stage": mvce_mc["avg_first_stage"],
                "avg_holding": mvce_mc["avg_holding"],
                "avg_stockout": mvce_mc["avg_stockout"],
                "avg_expiration": mvce_mc["avg_expiration"],
                "avg_second_stage": mvce_mc["avg_second_stage"],
                "avg_total": mvce_mc["avg_total"],
                "avg_stockout_rate": mvce_mc["avg_stockout_rate"],
                "algorithm_time": mvce_result["total_time"],
                "iter_count": mvce_result["iter_count"]
            }
            all_results.append(result_entry)

            # Print current repeat experiment results
            print(f"\n--- Repeat experiment {repeat + 1} Results ---")
            print(f"Average total cost: {mvce_mc['avg_total']:.2f}$")
            print(f"Average stockout rate: {mvce_mc['avg_stockout_rate']:.4f}%")

    # Save all raw results
    df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, "5.4_Experiment_Complete_Results.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nAll experiment results saved to: {output_path}")

    # Calculate average results per shelf life (corresponds to Table 5 in the paper)
    avg_df = df.groupby(["Mb"]).mean().reset_index()
    avg_output_path = os.path.join(output_dir, "5.4_Experiment_Average_Results.csv")
    avg_df.to_csv(avg_output_path, index=False, encoding="utf-8-sig")
    print(f"Average results saved to: {avg_output_path}")

    # Print final average results (exact format as Table 5 in the paper)
    print("\n" + "=" * 80)
    print("Section 5.4 Final Average Results (Corresponds to Table 5)")
    print("=" * 80)
    print(avg_df[[
        "Mb", "avg_first_stage", "avg_holding", "avg_stockout",
        "avg_expiration", "avg_second_stage", "avg_total"
    ]].round(2))

    # Print core conclusion preview
    print("\n" + "=" * 80)
    print("Core Conclusion Preview")
    print("=" * 80)
    min_cost_mb = avg_df.loc[avg_df["avg_total"].idxmin(), "Mb"]
    min_cost = avg_df["avg_total"].min()
    max_cost_mb = avg_df.loc[avg_df["avg_total"].idxmax(), "Mb"]
    max_cost = avg_df["avg_total"].max()
    cost_reduction = (max_cost - min_cost) / max_cost * 100

    print(f"Minimum average total cost: {min_cost:.2f}$ (Mb={min_cost_mb} days)")
    print(f"Maximum average total cost: {max_cost:.2f}$ (Mb={max_cost_mb} days)")
    print(f"Total cost reduction from optimal shelf life: {cost_reduction:.2f}%")
    print(f"Average stockout rate across all scenarios: {avg_df['avg_stockout_rate'].mean():.4f}%")

    # Verify paper's core conclusions
    print("\nPaper Core Conclusion Verification:")
    print(
        "1. Total cost decreases with shelf life extension: ✓" if avg_df["avg_total"].is_monotonic_decreasing else "✗")
    print("2. Expiration cost decreases with shelf life extension: ✓" if avg_df[
        "avg_expiration"].is_monotonic_decreasing else "✗")
    print("3. Holding cost increases with shelf life extension: ✓" if avg_df[
        "avg_holding"].is_monotonic_increasing else "✗")
    print("4. All scenarios have stockout rate < 0.15%: ✓" if (avg_df["avg_stockout_rate"] < 0.15).all() else "✗")

    return df, avg_df


# ==============================================================================
# ===================== Main Function (Run Section 5.4 Experiment) =====================
# ==============================================================================
def main():
    print("=" * 60)
    print("Paper Section 5.4 Experiment: Impact of Shelf Life on MVCE Model")
    print("Experiment Parameters: B=3, T=7, ρ=0.3, σ=0.5, Repeats=10")
    print("Shelf Life Gradient: 3, 4, 5, 6, 7 days")
    print("=" * 60)

    df, avg_df = run_experiment_54()

    print("\nExperiment completed successfully!")
    print("Result files saved to ./5.4_Experiment_Results/ directory")


if __name__ == "__main__":
    main()