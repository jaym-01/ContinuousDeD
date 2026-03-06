"""
Continuous-action critic networks for IQN and DQN.

Unlike the discrete variants in model.py (which output Q-values for every action),
these networks take (state, action) as joint input and output a scalar Q(s,a).

Architecture follows DSAC (Distributional Soft Actor-Critic):
  - State and action are concatenated and encoded via a shared linear layer.
  - For IQN: the state-action encoding is element-wise multiplied with a cosine
    quantile embedding (identical to the original IQN cosine trick), producing a
    scalar quantile estimate Z_τ(s,a).
  - For DQN: a plain 2-layer MLP producing a scalar Q(s,a).

Action selection in both cases is done externally by the agent via uniform
random shooting: sample K candidate actions, evaluate Q(s,a_i) for each,
return the argmax action.

References
----------
DSAC: https://arxiv.org/abs/2004.14547
IQN:  https://arxiv.org/abs/1806.06923
"""

import torch
import torch.nn as nn
import numpy as np


class ContinuousIQN(nn.Module):
    """IQN critic for continuous actions: Q(s,a) → distributional Z_τ(s,a).

    Parameters
    ----------
    state_size   : tuple, e.g. (4,) — flat observation shape
    action_dim   : int — number of continuous action dimensions
    layer_size   : int — hidden layer width
    seed         : int — random seed
    N            : int — number of quantile samples during training
    K            : int — number of quantile samples during action selection
    drm          : str — distortion risk measure: 'identity'|'cvar'|'cpw'|'power'
    eta          : float — DRM scaling factor
    device       : str
    """

    def __init__(
        self,
        state_size,
        action_dim,
        layer_size,
        seed,
        N,
        K=32,
        drm="identity",
        eta=0.71,
        device="cuda:0",
    ):
        super(ContinuousIQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_dim = state_size[0]
        self.action_dim = action_dim
        self.N = N
        self.K = K
        self.n_cos = 64
        self.layer_size = layer_size
        self.device = device
        self.drm = self._get_drm(drm, eta)
        # cos basis frequencies: shape (1, 1, n_cos)
        self.pis = torch.FloatTensor(
            [np.pi * i for i in range(1, self.n_cos + 1)]
        ).view(1, 1, self.n_cos).to(device)

        # State-action encoder: [s ∥ a] → layer_size
        self.head = nn.Linear(self.state_dim + self.action_dim, layer_size)
        # Cosine quantile embedding (same as original IQN)
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        # Output layers
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)

    def _get_drm(self, drm, eta=0.71):
        if drm.lower() == "cvar":
            f = lambda tau: tau * eta
        elif drm.lower() == "cpw":
            f = lambda tau: tau**eta / (tau**eta + (1 - tau)**eta) ** (1 / eta)
        elif drm.lower() == "identity":
            f = lambda tau: tau
        elif drm.lower() == "power":
            if eta <= 0:
                f = lambda tau: 1 - (1 - tau) ** (1 / (1 + abs(eta)))
            else:
                f = lambda tau: tau ** (1 / (1 + abs(eta)))
        else:
            raise ValueError(
                f"Unknown DRM: {drm!r}. Options: identity, cvar, cpw, power"
            )
        return np.vectorize(f)

    def calc_cos(self, batch_size, n_tau=8, use_drm=False):
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1)  # (batch, n_tau, 1)
        if use_drm:
            taus = torch.from_numpy(self.drm(taus.numpy())).float()
        cos = torch.cos(taus.to(self.device) * self.pis)  # (batch, n_tau, n_cos)
        return cos, taus.to(self.device)

    def forward(self, state, action, num_tau=8, use_drm=False):
        """
        Parameters
        ----------
        state   : (batch, state_dim)
        action  : (batch, action_dim)

        Returns
        -------
        quantiles : (batch, num_tau, 1)
        taus      : (batch, num_tau, 1)
        """
        batch_size = state.shape[0]

        # Encode state-action pair
        x = torch.relu(
            self.head(torch.cat([state, action], dim=-1))
        )  # (batch, layer_size)

        # Cosine quantile embedding
        cos, taus = self.calc_cos(batch_size, num_tau, use_drm)
        cos = cos.view(batch_size * num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(
            batch_size, num_tau, self.layer_size
        )  # (batch, num_tau, layer_size)

        # Element-wise multiply (DSAC-style feature fusion)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)  # (batch*num_tau, 1)

        return out.view(batch_size, num_tau, 1), taus

    def get_qvalue(self, state, action, use_drm=False):
        """Scalar Q(s,a) = mean of K quantile samples. Shape: (batch, 1)."""
        quantiles, _ = self.forward(state, action, self.K, use_drm)
        return quantiles.mean(dim=1)  # (batch, 1)


class ContinuousDQN(nn.Module):
    """Plain MLP critic for continuous actions: Q(s,a) → scalar.

    Parameters
    ----------
    state_size : tuple, e.g. (4,)
    action_dim : int
    layer_size : int
    seed       : int
    device     : str
    """

    def __init__(self, state_size, action_dim, layer_size, seed, device="cuda:0"):
        super(ContinuousDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_dim = state_size[0]
        self.action_dim = action_dim
        self.layer_size = layer_size

        self.head = nn.Linear(self.state_dim + self.action_dim, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, 1)

    def forward(self, state, action):
        """
        Parameters
        ----------
        state  : (batch, state_dim)
        action : (batch, action_dim)

        Returns
        -------
        q : (batch, 1)
        """
        x = torch.relu(self.head(torch.cat([state, action], dim=-1)))
        x = torch.relu(self.ff_1(x))
        return self.ff_2(x)  # (batch, 1)

    def get_qvalue(self, state, action, use_drm=False):
        """Returns (batch, 1). use_drm ignored (no quantile distribution)."""
        return self.forward(state, action)
