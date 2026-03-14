import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class BoundaryTracer:
    """Trace {a : Q_det(s,a) = τ} in [-0.5, 0.5]² using predictor-corrector.

    Parameters
    ----------
    net           : ContinuousIQN (must have get_qvalue_deterministic)
    state_tensor  : (state_dim,) or (1, state_dim) — detached, fixed state
    threshold     : scalar τ
    action_low    : float, lower action bound (same for both dims)
    action_high   : float, upper action bound
    device        : str
    """

    def __init__(self, net, state_tensor, threshold,
                 action_low=-0.5, action_high=0.5, device="cpu", q_fn=None):
        """
        q_fn : optional callable (state, action) -> (batch, 1) tensor.
               Defaults to net.get_qvalue_deterministic.  Use this to pass a
               CVaR-based Q-value function for CVaR boundary tracing.
        """
        self.net = net
        self.state = state_tensor.detach().to(device)
        if self.state.ndim == 1:
            self.state = self.state.unsqueeze(0)   # (1, state_dim)
        self.tau = float(threshold)
        self.a_low = float(action_low)
        self.a_high = float(action_high)
        self.device = device
        self.eval_count = 0
        self.q_fn = q_fn if q_fn is not None else net.get_qvalue_deterministic

    # Low-level evaluation helpers

    def _g_grad(self, a_np):
        """g(a) = Q_det(s,a) - τ and ∇_a g at a single point. Returns (float, ndarray)."""
        self.eval_count += 1
        a = torch.tensor(a_np, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)
        s = self.state  # (1, state_dim)
        q = self.q_fn(s, a)  # (1, 1)
        g = q[0, 0] - self.tau
        (grad,) = torch.autograd.grad(g, a, create_graph=False)
        return float(g.detach()), grad[0].detach().cpu().numpy()

    def _g_batch(self, actions_np):
        """Evaluate g and ∇g at N points in one forward-backward pass.

        Returns g_vals (N,) and grads (N, 2).  Batch-Jacobian trick: since
        each g_i depends only on a_i (independent batch items), differentiating
        sum(g_i) w.r.t. the full action tensor gives correct per-sample gradients.
        """
        N = len(actions_np)
        self.eval_count += N
        a = torch.tensor(actions_np, dtype=torch.float32, device=self.device).requires_grad_(True)
        s = self.state.expand(N, -1)
        q = self.q_fn(s, a)  # (N, 1)
        g = q[:, 0] - self.tau  # (N,)
        (grads,) = torch.autograd.grad(g.sum(), a, create_graph=False)
        return g.detach().cpu().numpy(), grads.detach().cpu().numpy()

    def _newton(self, a_np, max_iters=10, tol=1e-6):
        """Newton projection onto g=0.  Returns (point, converged)."""
        a = a_np.copy().astype(np.float32)
        for _ in range(max_iters):
            g, grad = self._g_grad(a)
            if abs(g) < tol:
                return a, True
            gn2 = float(np.dot(grad, grad))
            if gn2 < 1e-16:
                return a, False
            a = np.clip(a - (g / gn2) * grad, self.a_low, self.a_high)
        g, _ = self._g_grad(a)
        return a, abs(g) < tol

    
    # Phase 1: coarse grid with gradients
    def phase1(self, M=15):
        """Evaluate g and ∇g on an M×M grid. Returns (grid, g_vals, grads)."""
        pts = np.linspace(self.a_low, self.a_high, M, dtype=np.float32)
        xs, ys = np.meshgrid(pts, pts)
        grid = np.column_stack([xs.ravel(), ys.ravel()])  # (M², 2)
        g_vals, grads = self._g_batch(grid)
        return grid, g_vals, grads

    
    # Phase 2: seed detection
    def phase2(self, grid, g_vals, grads, M):
        """Find boundary seeds via sign-change bisection + gradient proximity."""
        seeds = []
        h_grid = (self.a_high - self.a_low) / (M - 1)

        # --- Mechanism A: sign-change bisection + Newton refinement ---
        for i in range(M):
            for j in range(M):
                idx = i * M + j
                for ni, nj in [(i, j + 1), (i + 1, j)]:
                    if ni >= M or nj >= M:
                        continue
                    nidx = ni * M + nj
                    if g_vals[idx] * g_vals[nidx] >= 0:
                        continue
                    # Bisection (7 iterations)
                    a_lo, a_hi = grid[idx].copy(), grid[nidx].copy()
                    g_lo = g_vals[idx]
                    for _ in range(7):
                        mid = (a_lo + a_hi) / 2
                        g_mid, _ = self._g_grad(mid)
                        if g_mid * g_lo < 0:
                            a_hi = mid
                        else:
                            a_lo, g_lo = mid, g_mid
                    # Newton polish
                    pt, ok = self._newton((a_lo + a_hi) / 2)
                    if ok:
                        seeds.append(pt)

        # --- Mechanism B: gradient-proximity Newton ---
        for idx in range(len(grid)):
            grad_norm = np.linalg.norm(grads[idx])
            if grad_norm < 1e-8:
                continue
            if abs(g_vals[idx]) / grad_norm < h_grid:
                pt, ok = self._newton(grid[idx].copy())
                if ok and self.a_low <= pt[0] <= self.a_high and self.a_low <= pt[1] <= self.a_high:
                    seeds.append(pt)

        return seeds

    
    # Phase 3: deduplication
    def phase3(self, seeds, h=0.01):
        """Greedy deduplication: keep seeds separated by at least 3h."""
        r = 3 * h
        kept = []
        for s in seeds:
            s = np.array(s, dtype=np.float32)
            if not kept or min(np.linalg.norm(s - k) for k in kept) >= r:
                kept.append(s)
        return kept

    
    # Phase 4: predictor-corrector tracing
    def _trace_one(self, start, direction, known_pts, max_steps=500):
        """Trace boundary in one direction from start. Returns list of 2-D points."""
        h = 0.01
        a_star = start.copy()
        path = []
        step = 0

        while step < max_steps:
            # ---- PREDICTOR ----
            g_val, grad = self._g_grad(a_star)
            gn = np.linalg.norm(grad)
            if gn < 1e-8:
                break
            tangent = np.array([-grad[1], grad[0]]) / gn * direction
            a_tilde = a_star + h * tangent
            if not (self.a_low <= a_tilde[0] <= self.a_high and
                    self.a_low <= a_tilde[1] <= self.a_high):
                break

            # ---- CORRECTOR (Newton) ----
            a_c = a_tilde.copy()
            converged, n_iters = False, 0
            for ni in range(10):
                gv, gd = self._g_grad(a_c)
                if abs(gv) < 1e-6:
                    converged, n_iters = True, ni + 1
                    break
                gn2 = float(np.dot(gd, gd))
                if gn2 < 1e-16:
                    break
                a_c = np.clip(a_c - (gv / gn2) * gd, self.a_low, self.a_high)

            if not converged:
                h /= 2
                if h < 1e-5:
                    break
                continue  # retry this step, don't increment

            # ---- DUPLICATE BOUNDARY CHECK ----
            if known_pts:
                kp = np.array(known_pts)
                if np.min(np.linalg.norm(kp - a_c, axis=1)) < 2 * h:
                    path.append(a_c.copy())
                    break

            # ---- CLOSED LOOP CHECK ----
            if len(path) > 10 and np.linalg.norm(a_c - start) < 2 * h:
                path.append(a_c.copy())
                break

            path.append(a_c.copy())
            a_star = a_c
            step += 1

            # Adaptive step size
            if n_iters == 1:
                h = min(h * 1.5, 0.05)
            elif n_iters >= 3:
                h = max(h * 0.5, 0.001)

        return path

    def phase4(self, seeds, known_pts=None):
        """Trace boundary from each seed in both directions."""
        if known_pts is None:
            known_pts = []
        components = []

        for seed in seeds:
            # Skip if this seed is already on a traced boundary
            if known_pts:
                kp = np.array(known_pts)
                if np.min(np.linalg.norm(kp - seed, axis=1)) < 3 * 0.01:
                    continue

            fwd = self._trace_one(seed, +1, known_pts)
            bwd = self._trace_one(seed, -1, known_pts)
            full = list(reversed(bwd)) + [seed] + fwd
            components.append(full)
            known_pts.extend(full)

        return components, known_pts

    
    # Phase 5: verification
    def phase5(self, components, known_pts, N_verify=100):
        """Check for missed boundary components using random verification points."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(self.a_low, self.a_high, (N_verify, 2)).astype(np.float32)
        g_vals, _ = self._g_batch(pts)

        delta = 2 * 0.01
        extra = []

        if known_pts:
            kp = np.array(known_pts)
            majority_sign = 1 if (g_vals > 0).mean() > 0.5 else -1
            for i, pt in enumerate(pts):
                if np.sign(g_vals[i]) == majority_sign:
                    continue
                # Minority-sign point far from traced boundaries → potential missed component
                if np.min(np.linalg.norm(kp - pt, axis=1)) > delta:
                    bp, ok = self._newton(pt)
                    if ok and np.min(np.linalg.norm(kp - bp, axis=1)) > delta:
                        extra.append(bp)

        if extra:
            extra = self.phase3(extra)
            new_comps, known_pts = self.phase4(extra, known_pts)
            components.extend(new_comps)

        return components, known_pts

    
    # Full pipeline
    def run(self, M=15):
        """Run all five phases.  Returns (components, debug_info)."""
        self.eval_count = 0

        grid, g_vals, grads = self.phase1(M)
        seeds_raw = self.phase2(grid, g_vals, grads, M)
        seeds = self.phase3(seeds_raw)
        components, known_pts = self.phase4(seeds)
        components, known_pts = self.phase5(components, known_pts)

        return components, {
            "grid": grid, "g_vals": g_vals, "grads": grads,
            "seeds_raw": seeds_raw, "seeds": seeds,
            "n_evals": self.eval_count,
        }