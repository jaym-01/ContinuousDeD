"""
Build a CVaR query function from a trained ContinuousIQN_Agent.

The returned callable maps a state (numpy array) to a scalar CVaR value,
suitable for use as the black-box function f in C2LSE (alse.py).

Typical usage
-------------
    from cvar_query import make_cvar_query_fn
    from alse import C2LSE

    # qd is a trained ContinuousIQN_Agent with sided_Q='negative'
    f = make_cvar_query_fn(qd, alpha=0.1)
    lse = C2LSE(f, bounds=..., threshold=-0.5)
    lse.run(50)
"""

import numpy as np
import torch


def make_cvar_query_fn(
    agent,
    alpha: float = 0.1,
    K_actions: int = 64,
    n_quantile_samples: int = 512,
    aggregation: str = "median",
):
    """
    Wrap a trained ContinuousIQN_Agent into a scalar CVaR query function.

    For each queried state the function:
      1. Samples K_actions random actions uniformly from the agent's action space.
      2. Evaluates n_quantile_samples quantiles of Z(s, a) for each action using
         the agent's local IQN network.
      3. Computes CVaR_alpha for each action — the expected value of the bottom-alpha
         fraction of quantile samples (expected worst-case outcome).
      4. Aggregates the per-action CVaR values with `aggregation`.

    This follows the DistDeD evaluation procedure (Killian et al., 2023):
    a state is at risk when the *median* CVaR over actions falls below a threshold.

    Parameters
    ----------
    agent : ContinuousIQN_Agent
        A trained agent, typically the D-network (sided_Q='negative').
        Must expose: qnetwork_local, action_low, action_high, action_dim, device.
    alpha : float
        CVaR confidence level in (0, 1].  Lower values are more conservative,
        considering only the worst-alpha fraction of the return distribution.
    K_actions : int
        Number of random actions sampled per state evaluation.
    n_quantile_samples : int
        Number of quantile samples drawn from the IQN per (state, action) pair.
        More samples give a more accurate CVaR estimate.
    aggregation : str
        How to combine per-action CVaR values. One of:
          'median' — median over actions (used in DistDeD for dead-end detection)
          'mean'   — mean over actions
          'max'    — best-case action (optimistic)
          'min'    — worst-case action (most pessimistic)

    Returns
    -------
    query_fn : callable
        query_fn(state: np.ndarray) -> float
        state shape: (state_dim,) or (1, state_dim)
    """
    if aggregation not in {"median", "mean", "max", "min"}:
        raise ValueError(
            f"aggregation must be one of median/mean/max/min, got {aggregation!r}"
        )

    device = agent.device
    action_low = torch.FloatTensor(agent.action_low).to(device)
    action_high = torch.FloatTensor(agent.action_high).to(device)

    # Number of quantile samples that form the CVaR tail
    n_cvar = max(1, int(alpha * n_quantile_samples))

    def query_fn(state: np.ndarray) -> float:
        """
        Parameters
        ----------
        state : np.ndarray of shape (state_dim,)

        Returns
        -------
        float — aggregated CVaR_alpha of Z(state, ·) over K random actions
        """
        state_np = np.atleast_2d(np.asarray(state, dtype=np.float32))  # (1, state_dim)
        state_t = torch.FloatTensor(state_np).to(device)                # (1, state_dim)

        # Sample K random actions uniformly within the agent's action bounds
        rand_u = torch.rand(K_actions, agent.action_dim, device=device)
        actions = rand_u * (action_high - action_low) + action_low  # (K, action_dim)

        # Tile the single state across all K actions
        state_exp = state_t.expand(K_actions, -1)  # (K, state_dim)

        agent.qnetwork_local.eval()
        with torch.no_grad():
            # Q_dist: (K, n_quantile_samples, 1) — distributional quantile values
            Q_dist, _ = agent.qnetwork_local(state_exp, actions, n_quantile_samples)
        agent.qnetwork_local.train()

        Q_dist = Q_dist.squeeze(-1)  # (K, n_quantile_samples)

        # CVaR_alpha = mean of the bottom-alpha fraction of sorted quantiles
        Q_sorted, _ = Q_dist.sort(dim=1)               # ascending: (K, n_quantile_samples)
        cvar_per_action = Q_sorted[:, :n_cvar].mean(dim=1)  # (K,)

        if aggregation == "median":
            result = cvar_per_action.median().item()
        elif aggregation == "mean":
            result = cvar_per_action.mean().item()
        elif aggregation == "max":
            result = cvar_per_action.max().item()
        else:  # min
            result = cvar_per_action.min().item()

        return result

    return query_fn
