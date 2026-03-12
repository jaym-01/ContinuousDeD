import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class C2LSE:
    """
    Confidence-based Continuous Level Set Estimation (C2LSE).

    Classifies points in a continuous domain S into:
      - Superlevel set  H = {x in S : f(x) > h}  — e.g. safe states
      - Sublevel set    L = {x in S : f(x) < h}  — e.g. dead-end states
      - Unclassified    M = {x in S : uncertain}

    Parameters
    ----------
    query_fn : callable
        f(x: np.ndarray) -> float.  The (expensive) black-box function to query.
        x has shape (state_dim,).
    bounds : array-like of shape (d, 2)
        [[lo_0, hi_0], ..., [lo_{d-1}, hi_{d-1}]] — search space bounds.
    threshold : float
        The level-set boundary h.
    epsilon : float
        Exploration parameter ε > 0 (Eq. 1).  Larger values promote global
        exploration; smaller values focus queries near the discovered boundary.
    beta : float
        Confidence multiplier for classification (β in Algorithm 1).
        A point x is classified as H if μ_T(x) − β σ_T(x) > h.
    noise : float
        Observation noise standard deviation for the GP (GP alpha = noise^2).
    n_random_candidates : int
        Number of random points sampled when maximising the acquisition function.
    n_restarts : int
        Number of additional L-BFGS-B restarts from random starting points.
    """

    def __init__(
        self,
        query_fn,
        bounds,
        threshold: float,
        epsilon: float = 0.1,
        beta: float = 2.0,
        noise: float = 1e-3,
        n_random_candidates: int = 1000,
        n_restarts: int = 5,
    ):
        self.query_fn = query_fn
        self.bounds = np.array(bounds, dtype=float)   # (d, 2)
        self.h = float(threshold)
        self.epsilon = float(epsilon)
        self.beta = float(beta)
        self.n_random_candidates = int(n_random_candidates)
        self.n_restarts = int(n_restarts)

        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise ** 2,
            normalize_y=True,
            n_restarts_optimizer=5,
        )

        self.X_obs: list = []   # list of 1-D float arrays, shape (d,)
        self.y_obs: list = []   # list of float scalars

    def _acquisition(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        mu, sigma = self.gp.predict(x, return_std=True)   # (n,), (n,)
        denom = np.maximum(self.epsilon, np.abs(mu - self.h))
        return sigma / denom

    def _random_point(self) -> np.ndarray:
        """Sample a uniformly random point from the search space."""
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest_next(self) -> np.ndarray:
        if len(self.X_obs) == 0:
            return self._random_point()

        d = self.bounds.shape[0]
        scipy_bounds = list(zip(self.bounds[:, 0], self.bounds[:, 1]))

        # --- Broad random search ---
        X_rand = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.n_random_candidates, d),
        )
        a_vals = self._acquisition(X_rand)           # (n_random_candidates,)
        best_idx = int(np.argmax(a_vals))
        best_x = X_rand[best_idx].copy()
        best_a = float(a_vals[best_idx])

        # --- Local optimisation from best random point + random restarts ---
        x0_list = [best_x] + [self._random_point() for _ in range(self.n_restarts - 1)]
        for x0 in x0_list:
            res = minimize(
                lambda x: -float(self._acquisition(x.reshape(1, -1))[0]),
                x0,
                method="L-BFGS-B",
                bounds=scipy_bounds,
            )
            if res.success and (-res.fun) > best_a:
                best_a = -res.fun
                best_x = res.x.copy()

        return best_x

    def query(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).flatten()
        y = float(self.query_fn(x))

        self.X_obs.append(x)
        self.y_obs.append(y)

        self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
        return y

    def classify(self, X_test: np.ndarray):
        if len(self.X_obs) == 0:
            raise RuntimeError("No observations yet. Call query() at least once first.")

        X_test = np.atleast_2d(X_test)
        mu, sigma = self.gp.predict(X_test, return_std=True)

        labels = np.full(len(X_test), "M", dtype=object)
        labels[mu - self.beta * sigma > self.h] = "H"   # superlevel (safe)
        labels[mu + self.beta * sigma < self.h] = "L"   # sublevel  (dead-end)

        return labels, mu, sigma

    def run(self, n_iterations: int, verbose: bool = True):
        for t in range(1, n_iterations + 1):
            x_next = self.suggest_next()
            y = self.query(x_next)

            if verbose:
                labels, mu, sigma = self.classify(np.atleast_2d(x_next))
                print(
                    f"[C2LSE {t:3d}/{n_iterations}] "
                    f"x={np.round(x_next, 3)}  "
                    f"f(x)={y:.4f}  "
                    f"μ={mu[0]:.4f}  σ={sigma[0]:.4f}  "
                    f"label={labels[0]}"
                )

        return np.array(self.X_obs), np.array(self.y_obs)
