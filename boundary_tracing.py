"""
Gradient-Informed Dead-End Boundary Tracing (Algorithm 1).

For a given encoded state s and a trained D-network (ContinuousIQN_OfflineAgent),
traces the zero-level-set of:

    g(s, a) = CVaR_alpha( Q_D(s, a) ) - delta_D

in 2-D action space using a predictor-corrector marching scheme, then computes
the dead-end volume fraction f_D(s) via the Shoelace formula.  A state is
classified as a dead-end when f_D(s) > 0.5.

Public API
----------
dead_end_volume_fraction(state, agent_dn, delta_D, alpha, ...) -> (f_D, polygons)
classify_dead_end(state, agent_dn, delta_D, alpha, ...) -> bool
"""

import numpy as np
import torch

# ---------------------------------------------------------------------------
# GPU-native helpers
# ---------------------------------------------------------------------------

def _eval_quantiles_batch(state_t, actions_t, agent_dn, num_tau):
    """Forward-only batch: return sorted quantiles for a batch of actions.

    state_t   : (1, state_dim) GPU tensor
    actions_t : (B, action_dim) GPU tensor
    Returns   : sorted_q (B, num_tau) float32 tensor on the same device
    """
    B = actions_t.shape[0]
    state_exp = state_t.expand(B, -1)
    with torch.no_grad():
        quantiles, _ = agent_dn.network(state_exp, actions_t, num_tau)  # (B, num_tau, 1)
    return quantiles.squeeze(-1).sort(dim=1)[0]  # (B, num_tau)


def _eval_g_batch(state_t, actions_t, agent_dn, delta_D, alpha, num_tau):
    """Forward-only batch evaluation of g(s, a) — no gradient, no backward.

    state_t   : (1, state_dim) GPU tensor
    actions_t : (B, action_dim) GPU tensor
    Returns   : g_vals (B,) float32 tensor on the same device
    """
    k = max(1, round(alpha * num_tau))
    sorted_q = _eval_quantiles_batch(state_t, actions_t, agent_dn, num_tau)
    cvar = sorted_q[:, :k].mean(dim=1)
    return cvar - delta_D


def _cvar_from_quantiles(sorted_q, alpha, delta_D):
    """Compute g = CVaR_alpha - delta_D from pre-sorted quantiles.

    sorted_q : (B, num_tau) tensor
    Returns  : (B,) tensor
    """
    k = max(1, round(alpha * sorted_q.shape[1]))
    return sorted_q[:, :k].mean(dim=1) - delta_D


def _compute_g_and_grad(state_t, action_t, agent_dn, delta_D, alpha, num_tau):
    """Evaluate g(s,a) and ∇_a g(s,a), keeping tensors on GPU.

    action_t must be a (1, action_dim) GPU tensor with requires_grad=True
    (created fresh each call so the graph is clean).

    Returns
    -------
    g_val  : float
    grad_a : (action_dim,) GPU tensor (detached)
    """
    quantiles, _ = agent_dn.network(state_t, action_t, num_tau)  # (1, num_tau, 1)
    k = max(1, round(alpha * num_tau))
    cvar = quantiles.squeeze(-1).sort(dim=1)[0][0, :k].mean()
    g = cvar - delta_D
    g.backward()
    return g.item(), action_t.grad.detach().squeeze(0)  # (action_dim,)


# ---------------------------------------------------------------------------
# Bisection (forward-only — no gradient needed)
# ---------------------------------------------------------------------------

def _bisect_to_boundary(a0_t, a1_t, g0, state_t, agent_dn, delta_D, alpha,
                        num_tau, eps_tol, max_iter=50):
    """Binary search along [a0_t, a1_t] for a* with |g(s,a*)| < eps_tol.

    a0_t, a1_t : (1, action_dim) GPU tensors
    g0         : float — g(a0) (pre-computed, sign known)
    Returns    : (1, action_dim) GPU tensor
    """
    g1 = _eval_g_batch(state_t, a1_t, agent_dn, delta_D, alpha, num_tau).item()
    for _ in range(max_iter):
        amid_t = 0.5 * (a0_t + a1_t)
        gmid = _eval_g_batch(state_t, amid_t, agent_dn, delta_D, alpha, num_tau).item()
        if abs(gmid) < eps_tol:
            return amid_t
        if gmid * g0 < 0:
            a1_t, g1 = amid_t, gmid
        else:
            a0_t, g0 = amid_t, gmid
    return 0.5 * (a0_t + a1_t)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rotate90_t(v_t):
    """Rotate a (action_dim,) GPU tensor 90° counter-clockwise."""
    return torch.stack([-v_t[1], v_t[0]])


def _shoelace_area(polygon):
    """Signed area of a 2-D polygon via the Shoelace formula."""
    p = np.asarray(polygon)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# Phase 1: Seed detection (GPU-batched grid scan)
# ---------------------------------------------------------------------------

def _phase1_seeds(state_t, agent_dn, delta_D, alpha, M, action_low, action_high,
                  eps_tol, eps_dup, num_tau):
    """Return a list of (action_dim,) numpy seed points on the g=0 boundary."""
    device = agent_dn.device
    a_lo_np = np.asarray(action_low, dtype=np.float32)
    a_hi_np = np.asarray(action_high, dtype=np.float32)
    a_lo_t  = torch.as_tensor(a_lo_np, device=device)
    a_hi_t  = torch.as_tensor(a_hi_np, device=device)

    # Build M×M grid on GPU
    ax0 = torch.linspace(float(a_lo_t[0]), float(a_hi_t[0]), M, device=device)
    ax1 = torch.linspace(float(a_lo_t[1]), float(a_hi_t[1]), M, device=device)
    G0, G1 = torch.meshgrid(ax0, ax1, indexing='ij')
    grid_t = torch.stack([G0.reshape(-1), G1.reshape(-1)], dim=1)  # (M*M, 2)

    # ---- Single batched forward pass for all grid g-values ----
    g_vals_t = _eval_g_batch(state_t, grid_t, agent_dn, delta_D, alpha, num_tau)
    g_vals   = g_vals_t.cpu().numpy()
    g_grid   = g_vals.reshape(M, M)

    ax0_np = ax0.cpu().numpy()
    ax1_np = ax1.cpu().numpy()

    seeds = []

    def _add_seed(a_t):
        a_np = a_t.squeeze(0).cpu().numpy()
        for s in seeds:
            if np.linalg.norm(a_np - s) < eps_dup:
                return
        seeds.append(np.clip(a_np, a_lo_np, a_hi_np))

    def _grid_pt(i, j):
        return torch.tensor([[ax0_np[i], ax1_np[j]]], dtype=torch.float32, device=device)

    # Sign-change edges → bisection (forward-only on GPU)
    for i in range(M):
        for j in range(M):
            g_ij = g_grid[i, j]
            if i + 1 < M and g_ij * g_grid[i + 1, j] < 0:
                a_star = _bisect_to_boundary(
                    _grid_pt(i, j), _grid_pt(i + 1, j), g_ij,
                    state_t, agent_dn, delta_D, alpha, num_tau, eps_tol
                )
                _add_seed(a_star)
            if j + 1 < M and g_ij * g_grid[i, j + 1] < 0:
                a_star = _bisect_to_boundary(
                    _grid_pt(i, j), _grid_pt(i, j + 1), g_ij,
                    state_t, agent_dn, delta_D, alpha, num_tau, eps_tol
                )
                _add_seed(a_star)

    # Near-boundary vertices → gradient Newton step (Algorithm 1, Phase 1).
    # Condition: |g(s,a)| / ‖∇_a g(s,a)‖ < δ_grid·√2/2
    # where δ_grid is the per-dimension grid spacing (max over dims for non-square grids).
    # δ_grid·√2/2 is the half-diagonal of one grid cell, i.e. the farthest a boundary
    # can be from a vertex and still be detected by the gradient step.
    delta_grid_np = (a_hi_np - a_lo_np) / (M - 1)
    delta_thresh  = float(np.max(delta_grid_np)) * np.sqrt(2) / 2   # Algorithm 1 exactly
    g_abs_max     = float(np.abs(g_vals).max()) + 1e-12
    # Loose prefilter: skip points where |g| > 5× delta_thresh (gradient would have
    # to be implausibly small to satisfy the condition).
    candidate_mask = np.abs(g_vals) < max(5.0 * delta_thresh, 0.1 * g_abs_max)

    for idx in np.where(candidate_mask)[0]:
        gv = float(g_vals[idx])
        a_t = grid_t[idx : idx + 1].clone().requires_grad_(True)
        _, gd_t = _compute_g_and_grad(state_t, a_t, agent_dn, delta_D, alpha, num_tau)
        # gd_t : (action_dim,)
        norm_gd = float(gd_t.norm())
        if norm_gd < 1e-12:
            continue
        if abs(gv) / norm_gd < delta_thresh:
            a_star_t = a_t.detach().squeeze(0) - (gv / norm_gd ** 2) * gd_t  # (action_dim,)
            a_star_t = torch.clamp(a_star_t, a_lo_t, a_hi_t)
            _add_seed(a_star_t)

    return seeds


# ---------------------------------------------------------------------------
# Phase 2: Predictor-corrector tracing (GPU-native inner loop)
# ---------------------------------------------------------------------------

def _phase2_trace(seed_np, state_t, agent_dn, delta_D, alpha, h0, eps_tol,
                  eps_close, eta, C_max, action_low, action_high, num_tau,
                  max_steps=2000):
    """Trace one connected boundary component starting from *seed_np*.

    Returns a (N, 2) numpy array of boundary points.
    """
    device = agent_dn.device
    a_lo_t = torch.as_tensor(np.asarray(action_low, np.float32), device=device)
    a_hi_t = torch.as_tensor(np.asarray(action_high, np.float32), device=device)
    a_lo_np = a_lo_t.cpu().numpy()
    a_hi_np = a_hi_t.cpu().numpy()

    seed_t = torch.tensor(seed_np, dtype=torch.float32, device=device)
    polygon_gpu = [seed_t.clone()]
    a_k = seed_t.clone()
    h = h0

    for _ in range(max_steps):
        a_in = a_k.unsqueeze(0).clone().requires_grad_(True)
        _, grad_k = _compute_g_and_grad(state_t, a_in, agent_dn, delta_D, alpha, num_tau)
        # grad_k : (action_dim,) GPU tensor
        norm_grad = float(grad_k.norm())
        if norm_grad < 1e-12:
            break

        # Predictor: step along tangent (90° rotation of normalised gradient)
        tangent = _rotate90_t(grad_k / norm_grad)
        a_pred = a_k + h * tangent

        # Corrector: Newton steps to project back onto g=0
        a_corr = a_pred.clone()
        c_count = 0
        for c in range(C_max):
            c_count = c + 1
            a_corr_in = a_corr.unsqueeze(0).clone().requires_grad_(True)
            g_corr, grad_corr = _compute_g_and_grad(
                state_t, a_corr_in, agent_dn, delta_D, alpha, num_tau
            )
            norm_gc = float(grad_corr.norm())
            if norm_gc < 1e-12:
                break
            a_corr = a_corr - (g_corr / norm_gc ** 2) * grad_corr
            if abs(g_corr) < eps_tol:
                break

        a_k = a_corr
        polygon_gpu.append(a_k.clone())

        if c_count >= C_max:
            h = h / eta
        elif c_count <= 2:
            h = h * eta
        h = float(np.clip(h, h0 / 10, h0 * 10))

        a_k_np = a_k.cpu().numpy()
        if len(polygon_gpu) > 3 and np.linalg.norm(a_k_np - seed_np) < eps_close:
            break
        if np.any(a_k_np < a_lo_np - 1e-3) or np.any(a_k_np > a_hi_np + 1e-3):
            break

    return np.stack([p.cpu().numpy() for p in polygon_gpu], axis=0)


# ---------------------------------------------------------------------------
# Phase 3: Classification
# ---------------------------------------------------------------------------

def _action_space_area(action_low, action_high):
    a_lo = np.asarray(action_low, dtype=np.float64)
    a_hi = np.asarray(action_high, dtype=np.float64)
    diffs = a_hi - a_lo
    area = float(np.prod(diffs[diffs > 0]))
    return max(area, 1e-12)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dead_end_volume_fraction_grid_batch(
    states,
    agent_dn,
    alphas,
    delta_D=-0.5,
    M=10,
    action_low=None,
    action_high=None,
    num_tau=64,
):
    """Grid-based f_D estimate for a batch of states — one forward pass for all.

    For N states and an M×M action grid, runs one forward pass of size N×M²,
    then computes CVaR at each alpha level from cached quantiles.

    Parameters
    ----------
    states : (N, state_dim) numpy array or GPU tensor

    Returns
    -------
    f_D : (N, len(alphas)) numpy array — f_D[i, j] for state i, alpha alphas[j]
    """
    device     = agent_dn.device
    action_dim = agent_dn.action_dim

    if action_low is None:
        action_low  = [0.0] * action_dim
    if action_high is None:
        action_high = [1.0] * action_dim

    states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
    N = states_t.shape[0]

    a_lo_t = torch.as_tensor(np.asarray(action_low,  np.float32), device=device)
    a_hi_t = torch.as_tensor(np.asarray(action_high, np.float32), device=device)

    ax0 = torch.linspace(float(a_lo_t[0]), float(a_hi_t[0]), M, device=device)
    ax1 = torch.linspace(float(a_lo_t[1]), float(a_hi_t[1]), M, device=device)
    G0, G1 = torch.meshgrid(ax0, ax1, indexing='ij')
    grid_t = torch.stack([G0.reshape(-1), G1.reshape(-1)], dim=1)  # (M², 2)
    A = grid_t.shape[0]  # M²

    # Expand: each state paired with every action → (N×M², state_dim/action_dim)
    states_exp  = states_t.unsqueeze(1).expand(N, A, -1).reshape(N * A, -1)
    actions_exp = grid_t.unsqueeze(0).expand(N, A, -1).reshape(N * A, -1)

    agent_dn.eval()
    with torch.no_grad():
        quantiles, _ = agent_dn.network(states_exp, actions_exp, num_tau)  # (N*A, num_tau, 1)

    sorted_q = quantiles.squeeze(-1).sort(dim=1)[0].reshape(N, A, num_tau)  # (N, A, num_tau)

    alphas = list(alphas)
    f_D = np.empty((N, len(alphas)), dtype=np.float32)
    for j, alpha in enumerate(alphas):
        k = max(1, round(alpha * num_tau))
        cvar = sorted_q[:, :, :k].mean(dim=2)  # (N, A)
        g = cvar - delta_D                      # (N, A)
        # g < 0 ↔ CVaR_alpha(Q_D) < delta_D — the dead-end region.
        # The predictor-corrector traces the g=0 boundary CCW, enclosing g < 0,
        # so f_D = fraction of dead-end actions = fraction where g < 0.
        f_D[:, j] = (g < 0).float().mean(dim=1).cpu().numpy()

    return f_D


def grid_cvar_batch(
    states,
    agent,
    alphas,
    M=10,
    action_low=None,
    action_high=None,
    num_tau=64,
    agg='max',
):
    """Grid-based CVaR aggregation over actions for a batch of states.

    Identical forward pass to dead_end_volume_fraction_grid_batch, but instead
    of computing the fraction of actions below a fixed delta threshold it returns
    the aggregated (max or mean) CVaR directly. This means no delta_D calibration
    is needed: sweeping the downstream flagging threshold is equivalent to sweeping
    delta_D across the full Q-value range.

    Parameters
    ----------
    states    : (N, state_dim) array
    agent     : ContinuousIQN_OfflineAgent
    alphas    : sequence of CVaR risk levels in (0, 1]
    M         : grid side length (M×M actions evaluated per state)
    agg       : 'max'  → best-case action (pessimistic dead-end view)
                'mean' → average over grid

    Returns
    -------
    result : (N, len(alphas)) float32
        result[i, j] = agg_a CVaR_{alphas[j]}(Q(states[i], a))
    """
    device     = agent.device
    action_dim = agent.action_dim

    if action_low is None:
        action_low  = [0.0] * action_dim
    if action_high is None:
        action_high = [1.0] * action_dim

    states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=device)
    N = states_t.shape[0]

    a_lo_t = torch.as_tensor(np.asarray(action_low,  np.float32), device=device)
    a_hi_t = torch.as_tensor(np.asarray(action_high, np.float32), device=device)

    ax0 = torch.linspace(float(a_lo_t[0]), float(a_hi_t[0]), M, device=device)
    ax1 = torch.linspace(float(a_lo_t[1]), float(a_hi_t[1]), M, device=device)
    G0, G1 = torch.meshgrid(ax0, ax1, indexing='ij')
    grid_t = torch.stack([G0.reshape(-1), G1.reshape(-1)], dim=1)  # (M², 2)
    A = grid_t.shape[0]

    states_exp  = states_t.unsqueeze(1).expand(N, A, -1).reshape(N * A, -1)
    actions_exp = grid_t.unsqueeze(0).expand(N, A, -1).reshape(N * A, -1)

    agent.eval()
    with torch.no_grad():
        quantiles, _ = agent.network(states_exp, actions_exp, num_tau)  # (N*A, num_tau, 1)

    sorted_q = quantiles.squeeze(-1).sort(dim=1)[0].reshape(N, A, num_tau)  # (N, A, num_tau)

    alphas = list(alphas)
    result = np.empty((N, len(alphas)), dtype=np.float32)
    for j, alpha in enumerate(alphas):
        k = max(1, round(alpha * num_tau))
        cvar = sorted_q[:, :, :k].mean(dim=2)  # (N, A)
        if agg == 'max':
            result[:, j] = cvar.max(dim=1)[0].cpu().numpy()
        else:
            result[:, j] = cvar.mean(dim=1).cpu().numpy()

    return result


def dead_end_volume_fraction_grid(
    state,
    agent_dn,
    alphas,
    delta_D=-0.5,
    M=10,
    action_low=None,
    action_high=None,
    num_tau=64,
):
    """Fast grid-based estimate of f_D for multiple alpha levels.

    Replaces the predictor-corrector boundary tracer with a single batched
    forward pass over an M×M action grid.  f_D is the fraction of grid cells
    where g(s,a) = CVaR_alpha(Q_D(s,a)) - delta_D > 0.

    One forward pass is shared across all alpha values (quantiles are computed
    once; CVaR is sliced per alpha).  Complexity: O(1) network calls vs the
    tracer's O(max_steps × C_max) sequential calls per state.

    Parameters
    ----------
    alphas : sequence of float
    M      : int — grid side length; M=10 gives 100 actions per forward pass

    Returns
    -------
    dict {float(alpha): float f_D}
    """
    device     = agent_dn.device
    action_dim = agent_dn.action_dim

    if action_low is None:
        action_low  = [0.0] * action_dim
    if action_high is None:
        action_high = [1.0] * action_dim

    state_t = torch.tensor(state[None], dtype=torch.float32, device=device)

    a_lo_t = torch.as_tensor(np.asarray(action_low,  np.float32), device=device)
    a_hi_t = torch.as_tensor(np.asarray(action_high, np.float32), device=device)

    ax0 = torch.linspace(float(a_lo_t[0]), float(a_hi_t[0]), M, device=device)
    ax1 = torch.linspace(float(a_lo_t[1]), float(a_hi_t[1]), M, device=device)
    G0, G1 = torch.meshgrid(ax0, ax1, indexing='ij')
    grid_t = torch.stack([G0.reshape(-1), G1.reshape(-1)], dim=1)  # (M*M, 2)

    agent_dn.eval()
    sorted_q = _eval_quantiles_batch(state_t, grid_t, agent_dn, num_tau)  # (M*M, num_tau)

    result = {}
    for alpha in alphas:
        g_vals = _cvar_from_quantiles(sorted_q, alpha, delta_D)  # (M*M,)
        result[float(alpha)] = float((g_vals < 0).float().mean())

    return result


def dead_end_volume_fraction(
    state,
    agent_dn,
    delta_D=-0.5,
    alpha=0.1,
    M=5,
    h0=0.05,
    eps_tol=1e-4,
    eps_close=0.02,
    eps_dup=0.02,
    eta=1.5,
    C_max=10,
    action_low=None,
    action_high=None,
    num_tau=64,
):
    """Compute f_D(s) and the traced boundary polygons for a single state.

    Parameters
    ----------
    state      : (state_dim,) numpy array — NCDE-encoded state
    agent_dn   : ContinuousIQN_OfflineAgent (D-network, sided_Q='negative')
    delta_D    : float — g = CVaR_alpha(Q_D(s,a)) - delta_D; boundary is g=0
    alpha      : float — CVaR level (0,1]; smaller = more pessimistic
    M          : int — grid resolution for Phase 1 seed detection
    h0         : float — initial predictor step size
    eps_tol    : float — corrector convergence tolerance on |g|
    eps_close  : float — distance to seed that closes a traced loop
    eps_dup    : float — min distance between distinct seeds
    eta        : float — step-size adaptation factor (>1)
    C_max      : int — max corrector iterations per step
    action_low : list[float] or None — lower bounds on action space; defaults to [0,0]
    action_high: list[float] or None — upper bounds on action space; defaults to [1,1]
    num_tau    : int — IQN quantile samples for CVaR estimation

    Returns
    -------
    f_D      : float — fraction of action space classified as dead-end
    polygons : list of (N_i, 2) numpy arrays — one per traced boundary component
    """
    device = agent_dn.device
    action_dim = agent_dn.action_dim

    if action_low is None:
        action_low  = [0.0] * action_dim
    if action_high is None:
        action_high = [1.0] * action_dim

    state_t = torch.tensor(state[None], dtype=torch.float32, device=device)

    agent_dn.eval()
    with torch.enable_grad():
        # Phase 1: seed detection (batched grid scan)
        seeds = _phase1_seeds(
            state_t, agent_dn, delta_D, alpha, M,
            action_low, action_high, eps_tol, eps_dup, num_tau
        )

        if not seeds:
            mid_t = torch.tensor(
                [0.5 * (lo + hi) for lo, hi in zip(action_low, action_high)],
                dtype=torch.float32, device=device
            ).unsqueeze(0)
            g_mid = _eval_g_batch(state_t, mid_t, agent_dn, delta_D, alpha, num_tau).item()
            # g < 0 everywhere → whole action space is dead-end → f_D = 1
            # g > 0 everywhere → no dead-end actions → f_D = 0
            f_D = 0.0 if g_mid > 0 else 1.0
            return f_D, []

        # Phase 2: predictor-corrector tracing
        claimed = [False] * len(seeds)
        polygons = []
        for si, seed in enumerate(seeds):
            if claimed[si]:
                continue
            polygon = _phase2_trace(
                seed, state_t, agent_dn, delta_D, alpha, h0, eps_tol,
                eps_close, eta, C_max, action_low, action_high, num_tau
            )
            if len(polygon) < 3:
                continue
            polygons.append(polygon)
            for sj, s in enumerate(seeds):
                if not claimed[sj]:
                    for pt in polygon:
                        if np.linalg.norm(s - pt) < eps_dup:
                            claimed[sj] = True
                            break

        # Phase 3: volume fraction via Shoelace
        total_area = _action_space_area(action_low, action_high)
        dead_area  = sum(_shoelace_area(p) for p in polygons)
        f_D = min(dead_area / total_area, 1.0)

    return f_D, polygons


def dead_end_volume_fraction_multi_alpha(
    state,
    agent_dn,
    alphas,
    delta_D=-0.5,
    M=5,
    h0=0.05,
    eps_tol=1e-4,
    eps_close=0.02,
    eps_dup=0.02,
    eta=1.5,
    C_max=10,
    action_low=None,
    action_high=None,
    num_tau=64,
):
    """Compute f_D(s) for multiple CVaR levels in a single pass over the state.

    The M×M grid forward pass is shared across all alpha values — quantiles are
    computed once and CVaR is sliced per alpha, reducing network calls by len(alphas)×
    for the grid scan.  Bisection and tracing are still done per-alpha as the
    boundary location changes with alpha.

    Parameters
    ----------
    alphas : sequence of float — CVaR levels to evaluate

    Returns
    -------
    f_D_per_alpha : dict {alpha: f_D} — volume fraction for each alpha
    """
    device = agent_dn.device
    action_dim = agent_dn.action_dim

    if action_low is None:
        action_low  = [0.0] * action_dim
    if action_high is None:
        action_high = [1.0] * action_dim

    a_lo_np = np.asarray(action_low, dtype=np.float32)
    a_hi_np = np.asarray(action_high, dtype=np.float32)
    a_lo_t  = torch.as_tensor(a_lo_np, device=device)
    a_hi_t  = torch.as_tensor(a_hi_np, device=device)

    state_t = torch.tensor(state[None], dtype=torch.float32, device=device)

    # Build M×M grid on GPU (shared across all alphas)
    ax0 = torch.linspace(float(a_lo_t[0]), float(a_hi_t[0]), M, device=device)
    ax1 = torch.linspace(float(a_lo_t[1]), float(a_hi_t[1]), M, device=device)
    G0, G1 = torch.meshgrid(ax0, ax1, indexing='ij')
    grid_t = torch.stack([G0.reshape(-1), G1.reshape(-1)], dim=1)  # (M*M, 2)

    ax0_np = ax0.cpu().numpy()
    ax1_np = ax1.cpu().numpy()
    delta_grid_np = (a_hi_np - a_lo_np) / (M - 1)

    agent_dn.eval()

    # ---- Single shared grid forward pass ----
    sorted_q = _eval_quantiles_batch(state_t, grid_t, agent_dn, num_tau)  # (M*M, num_tau)

    total_area = _action_space_area(action_low, action_high)
    f_D_per_alpha = {}

    with torch.enable_grad():
        for alpha in alphas:
            # Compute g-values for this alpha from cached quantiles
            g_vals_t = _cvar_from_quantiles(sorted_q, alpha, delta_D)
            g_vals   = g_vals_t.cpu().numpy()
            g_grid   = g_vals.reshape(M, M)

            seeds = []
            delta_thresh = float(np.max(delta_grid_np)) * np.sqrt(2) / 2  # Algorithm 1 exactly

            def _add_seed(a_t):
                a_np = a_t.detach().cpu().numpy().reshape(-1)
                for s in seeds:
                    if np.linalg.norm(a_np - s) < eps_dup:
                        return
                seeds.append(np.clip(a_np, a_lo_np, a_hi_np))

            def _grid_pt(i, j):
                return torch.tensor([[ax0_np[i], ax1_np[j]]], dtype=torch.float32, device=device)

            # Sign-change edges → bisection
            for i in range(M):
                for j in range(M):
                    g_ij = g_grid[i, j]
                    if i + 1 < M and g_ij * g_grid[i + 1, j] < 0:
                        _add_seed(_bisect_to_boundary(
                            _grid_pt(i, j), _grid_pt(i + 1, j), g_ij,
                            state_t, agent_dn, delta_D, alpha, num_tau, eps_tol
                        ))
                    if j + 1 < M and g_ij * g_grid[i, j + 1] < 0:
                        _add_seed(_bisect_to_boundary(
                            _grid_pt(i, j), _grid_pt(i, j + 1), g_ij,
                            state_t, agent_dn, delta_D, alpha, num_tau, eps_tol
                        ))

            # Near-boundary Newton step (gradient only for candidates)
            g_abs_max    = float(np.abs(g_vals).max()) + 1e-12
            candidate_mask = np.abs(g_vals) < max(5.0 * delta_thresh, 0.1 * g_abs_max)
            for idx in np.where(candidate_mask)[0]:
                gv  = float(g_vals[idx])
                a_t = grid_t[idx : idx + 1].clone().requires_grad_(True)
                _, gd_t = _compute_g_and_grad(state_t, a_t, agent_dn, delta_D, alpha, num_tau)
                norm_gd = float(gd_t.norm())
                if norm_gd < 1e-12:
                    continue
                if abs(gv) / norm_gd < delta_thresh:
                    a_star = a_t.detach().squeeze(0) - (gv / norm_gd ** 2) * gd_t
                    _add_seed(torch.clamp(a_star, a_lo_t, a_hi_t))

            if not seeds:
                mid_t = torch.tensor(
                    [0.5 * (lo + hi) for lo, hi in zip(action_low, action_high)],
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                g_mid = _eval_g_batch(state_t, mid_t, agent_dn, delta_D, alpha, num_tau).item()
                f_D_per_alpha[float(alpha)] = 0.0 if g_mid > 0 else 1.0
                continue

            claimed = [False] * len(seeds)
            polygons = []
            for si, seed in enumerate(seeds):
                if claimed[si]:
                    continue
                polygon = _phase2_trace(
                    seed, state_t, agent_dn, delta_D, alpha, h0, eps_tol,
                    eps_close, eta, C_max, action_low, action_high, num_tau
                )
                if len(polygon) < 3:
                    continue
                polygons.append(polygon)
                for sj, s in enumerate(seeds):
                    if not claimed[sj]:
                        for pt in polygon:
                            if np.linalg.norm(s - pt) < eps_dup:
                                claimed[sj] = True
                                break

            dead_area = sum(_shoelace_area(p) for p in polygons)
            f_D_per_alpha[float(alpha)] = min(dead_area / total_area, 1.0)

    return f_D_per_alpha


def classify_dead_end(
    state,
    agent_dn,
    threshold=0.5,
    delta_D=-0.5,
    alpha=0.1,
    **kwargs,
):
    """Return True if state s is a dead-end (f_D(s) > threshold).

    Parameters
    ----------
    state     : (state_dim,) numpy array
    agent_dn  : ContinuousIQN_OfflineAgent
    threshold : float — volume fraction above which state is a dead-end (default 0.5)
    delta_D   : float — g-function threshold
    alpha     : float — CVaR level
    **kwargs  : forwarded to dead_end_volume_fraction

    Returns
    -------
    is_dead_end : bool
    f_D         : float — the computed volume fraction
    """
    f_D, _ = dead_end_volume_fraction(state, agent_dn, delta_D=delta_D, alpha=alpha, **kwargs)
    return f_D > threshold, f_D
