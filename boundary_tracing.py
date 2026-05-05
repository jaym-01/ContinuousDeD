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
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_g_and_grad(state_t, action_np, agent_dn, delta_D, alpha, num_tau):
    """Evaluate g(s,a) and ∇_a g(s,a).

    Parameters
    ----------
    state_t   : (1, state_dim) float tensor on agent device, no grad needed
    action_np : (action_dim,) numpy array
    agent_dn  : ContinuousIQN_OfflineAgent
    delta_D   : float, dead-end threshold (e.g. -0.5 for D-net in [-1,0])
    alpha     : float, CVaR level in (0, 1]
    num_tau   : int, quantile samples

    Returns
    -------
    g_val  : float
    grad_a : (action_dim,) numpy array  — ∇_a g
    """
    device = agent_dn.device
    action_t = torch.tensor(
        action_np[None], dtype=torch.float32, device=device, requires_grad=True
    )

    quantiles, _ = agent_dn.network(state_t, action_t, num_tau)  # (1, num_tau, 1)
    quantiles_sorted = quantiles.squeeze().sort()[0]              # (num_tau,)

    k = max(1, round(alpha * num_tau))
    cvar = quantiles_sorted[:k].mean()
    g = cvar - delta_D

    g.backward()
    grad_a = action_t.grad.detach().cpu().numpy().ravel()
    return g.item(), grad_a


def _bisect_to_boundary(a0, a1, g0, state_t, agent_dn, delta_D, alpha, num_tau, eps_tol, max_iter=50):
    """Binary search along edge [a0, a1] for a* with |g(s,a*)| < eps_tol.

    Assumes g0 = g(a0) and g1 = g(a1) have opposite signs.
    """
    g1, _ = _compute_g_and_grad(state_t, a1, agent_dn, delta_D, alpha, num_tau)
    for _ in range(max_iter):
        amid = 0.5 * (a0 + a1)
        gmid, _ = _compute_g_and_grad(state_t, amid, agent_dn, delta_D, alpha, num_tau)
        if abs(gmid) < eps_tol:
            return amid
        if gmid * g0 < 0:
            a1, g1 = amid, gmid
        else:
            a0, g0 = amid, gmid
    return 0.5 * (a0 + a1)


def _rotate90(v):
    """Rotate 2-D vector 90° counter-clockwise."""
    return np.array([-v[1], v[0]])


def _shoelace_area(polygon):
    """Signed area of a 2-D polygon via the Shoelace formula."""
    p = np.asarray(polygon)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# Phase 1: Seed detection
# ---------------------------------------------------------------------------

def _phase1_seeds(state_t, agent_dn, delta_D, alpha, M, action_low, action_high,
                  eps_tol, eps_dup, num_tau):
    """Return a list of (action_dim,) seed points on the g=0 boundary."""
    a_lo = np.asarray(action_low, dtype=np.float32)
    a_hi = np.asarray(action_high, dtype=np.float32)

    # Build M×M grid over 2-D action space
    ax0 = np.linspace(a_lo[0], a_hi[0], M)
    ax1 = np.linspace(a_lo[1], a_hi[1], M)
    G0, G1 = np.meshgrid(ax0, ax1, indexing='ij')              # (M, M)
    grid = np.stack([G0, G1], axis=-1).reshape(-1, 2)          # (M*M, 2)

    # --- Evaluate g and ∇g on every grid vertex ---
    g_vals = np.zeros(len(grid), dtype=np.float32)
    grad_vals = np.zeros((len(grid), 2), dtype=np.float32)
    for idx, a in enumerate(grid):
        gv, gd = _compute_g_and_grad(state_t, a, agent_dn, delta_D, alpha, num_tau)
        g_vals[idx] = gv
        grad_vals[idx] = gd
    g_grid = g_vals.reshape(M, M)

    seeds = []

    def _add_seed(a_star):
        """Add seed if not a duplicate."""
        for s in seeds:
            if np.linalg.norm(a_star - s) < eps_dup:
                return
        # Clamp to action box
        a_clamped = np.clip(a_star, a_lo, a_hi)
        seeds.append(a_clamped)

    # --- Sign-change edges → bisection ---
    delta_grid = (a_hi - a_lo) / (M - 1)  # per-axis cell width

    def _grid_pt(i, j):
        return np.array([ax0[i], ax1[j]], dtype=np.float32)

    for i in range(M):
        for j in range(M):
            idx_ij = i * M + j
            g_ij = g_grid[i, j]
            # Horizontal edge to (i+1, j)
            if i + 1 < M:
                g_next = g_grid[i + 1, j]
                if g_ij * g_next < 0:
                    a_star = _bisect_to_boundary(
                        _grid_pt(i, j), _grid_pt(i + 1, j), g_ij,
                        state_t, agent_dn, delta_D, alpha, num_tau, eps_tol
                    )
                    _add_seed(a_star)
            # Vertical edge to (i, j+1)
            if j + 1 < M:
                g_next = g_grid[i, j + 1]
                if g_ij * g_next < 0:
                    a_star = _bisect_to_boundary(
                        _grid_pt(i, j), _grid_pt(i, j + 1), g_ij,
                        state_t, agent_dn, delta_D, alpha, num_tau, eps_tol
                    )
                    _add_seed(a_star)

    # --- Near-boundary vertices → gradient Newton step ---
    delta_thresh = np.linalg.norm(delta_grid) * np.sqrt(2) / 2  # δ_grid √2 / 2
    for idx, (a, gv, gd) in enumerate(zip(grid, g_vals, grad_vals)):
        norm_gd = np.linalg.norm(gd)
        if norm_gd < 1e-12:
            continue
        if abs(gv) / norm_gd < delta_thresh:
            a_star = a - (gv / (norm_gd ** 2)) * gd
            a_star = np.clip(a_star, a_lo, a_hi)
            _add_seed(a_star)

    return seeds


# ---------------------------------------------------------------------------
# Phase 2: Predictor-corrector tracing
# ---------------------------------------------------------------------------

def _phase2_trace(seed, state_t, agent_dn, delta_D, alpha, h0, eps_tol,
                  eps_close, eta, C_max, action_low, action_high, num_tau,
                  max_steps=2000):
    """Trace one connected boundary component starting from *seed*.

    Returns a (N, 2) numpy array of boundary points (the polygon).
    """
    a_lo = np.asarray(action_low, dtype=np.float32)
    a_hi = np.asarray(action_high, dtype=np.float32)

    polygon = [seed.copy()]
    a_k = seed.copy()
    h = h0

    for _ in range(max_steps):
        _, grad_k = _compute_g_and_grad(state_t, a_k, agent_dn, delta_D, alpha, num_tau)
        norm_grad = np.linalg.norm(grad_k)
        if norm_grad < 1e-12:
            break

        # Predictor: step along tangent direction (rotate gradient 90°)
        tangent = _rotate90(grad_k / norm_grad)
        a_pred = a_k + h * tangent

        # Corrector: Newton steps to project back onto g=0
        a_corr = a_pred.copy()
        c_count = 0
        for c in range(C_max):
            c_count = c + 1
            g_corr, grad_corr = _compute_g_and_grad(
                state_t, a_corr, agent_dn, delta_D, alpha, num_tau
            )
            norm_gc = np.linalg.norm(grad_corr)
            if norm_gc < 1e-12:
                break
            a_corr = a_corr - (g_corr / (norm_gc ** 2)) * grad_corr
            if abs(g_corr) < eps_tol:
                break

        a_next = a_corr
        polygon.append(a_next.copy())

        # Adaptive step size: shrink if corrector ran many iterations, grow otherwise
        if c_count >= C_max:
            h = h / eta
        elif c_count <= 2:
            h = h * eta
        h = float(np.clip(h, h0 / 10, h0 * 10))

        # Termination: loop closed or left action space
        if len(polygon) > 3 and np.linalg.norm(a_next - seed) < eps_close:
            break
        if np.any(a_next < a_lo - 1e-3) or np.any(a_next > a_hi + 1e-3):
            break

        a_k = a_next

    return np.array(polygon)


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

def dead_end_volume_fraction(
    state,
    agent_dn,
    delta_D=-0.5,
    alpha=0.1,
    M=20,
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
        # Phase 1: seed detection
        seeds = _phase1_seeds(
            state_t, agent_dn, delta_D, alpha, M,
            action_low, action_high, eps_tol, eps_dup, num_tau
        )

        if not seeds:
            # No boundary found — check if the entire action space is inside g>0 or g<0
            mid = 0.5 * (np.asarray(action_low) + np.asarray(action_high))
            g_mid, _ = _compute_g_and_grad(state_t, mid, agent_dn, delta_D, alpha, num_tau)
            f_D = 1.0 if g_mid > 0 else 0.0
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
            # Mark seeds that fall inside this polygon as claimed
            for sj, s in enumerate(seeds):
                if not claimed[sj]:
                    for pt in polygon:
                        if np.linalg.norm(s - pt) < eps_dup:
                            claimed[sj] = True
                            break

        # Phase 3: volume fraction via Shoelace
        total_area = _action_space_area(action_low, action_high)
        dead_area = sum(_shoelace_area(p) for p in polygons)
        f_D = min(dead_area / total_area, 1.0)

    return f_D, polygons


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
