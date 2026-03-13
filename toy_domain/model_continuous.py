"""
DSAC: https://arxiv.org/abs/2004.14547
IQN:  https://arxiv.org/abs/1806.06923
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _DRM:
    """Picklable distortion risk measure callable."""
    def __init__(self, name, eta):
        self.name = name.lower()
        self.eta = eta

    def __call__(self, tau):
        eta = self.eta
        if self.name == 'cvar':
            return tau * eta
        elif self.name == 'cpw':
            return tau**eta / (tau**eta + (1 - tau)**eta)**(1 / eta)
        elif self.name == 'identity':
            return tau
        elif self.name == 'power':
            if eta <= 0:
                return 1 - (1 - tau)**(1 / (1 + abs(eta)))
            else:
                return tau**(1 / (1 + abs(eta)))
        else:
            raise ValueError(f"Unknown DRM: {self.name!r}. Options: identity, cvar, cpw, power")


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
        state_low=None,
        state_high=None,
        action_low=None,
        action_high=None,
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

        # Normalisation: map state and action to ~ [-1, 1]
        if state_low is not None and state_high is not None:
            s_low = np.array(state_low, dtype=np.float32)
            s_high = np.array(state_high, dtype=np.float32)
            s_mean = (s_high + s_low) / 2.0
            s_scale = (s_high - s_low) / 2.0
        else:  # SpaceEnv defaults: (x,y,vx,vy), map 10×10, vel range ±2
            s_mean = np.array([5.0, 5.0, 0.0, 0.0], dtype=np.float32)
            s_scale = np.array([5.0, 5.0, 2.0, 2.0], dtype=np.float32)

        if action_low is not None and action_high is not None:
            a_low = np.array(action_low, dtype=np.float32)
            a_high = np.array(action_high, dtype=np.float32)
            a_mean = (a_high + a_low) / 2.0
            a_scale = (a_high - a_low) / 2.0
        else:  # SpaceEnv defaults: thrust ∈ [-0.5, 0.5]
            a_mean = np.zeros(action_dim, dtype=np.float32)
            a_scale = np.full(action_dim, 0.5, dtype=np.float32)

        self._s_mean = torch.FloatTensor(s_mean)
        self._s_scale = torch.FloatTensor(s_scale)
        self._a_mean = torch.FloatTensor(a_mean)
        self._a_scale = torch.FloatTensor(a_scale)
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
        return _DRM(drm, eta)

    def calc_cos(self, batch_size, n_tau=8, use_drm=False):
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1)  # (batch, n_tau, 1)
        if use_drm:
            taus = torch.from_numpy(self.drm(taus.numpy())).float()
        cos = torch.cos(taus.to(self.device) * self.pis)  # (batch, n_tau, n_cos)
        return cos, taus.to(self.device)

    def _normalise(self, state, action):
        """Normalise (state, action) to ~ [-1, 1] per dimension."""
        dev = state.device
        s = (state  - self._s_mean.to(dev))  / self._s_scale.to(dev)
        a = (action - self._a_mean.to(dev))  / self._a_scale.to(dev)
        return torch.cat([s, a], dim=-1)

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

        # Encode normalised state-action pair
        x = torch.relu(
            self.head(self._normalise(state, action))
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


class GaussianActor(nn.Module):
    """Stochastic Gaussian actor for continuous action spaces (DSAC).

    Outputs a squashed Gaussian policy: actions are sampled via the
    reparameterisation trick, then passed through tanh and scaled to [low, high].

    Parameters
    ----------
    state_size  : tuple, e.g. (4,)
    action_dim  : int
    layer_size  : int — hidden layer width
    action_low  : array-like — per-dimension lower bounds
    action_high : array-like — per-dimension upper bounds
    seed        : int
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_size, action_dim, layer_size, action_low, action_high, seed):
        super(GaussianActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        state_dim = state_size[0]

        self.head = nn.Linear(state_dim, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.mean_head = nn.Linear(layer_size, action_dim)
        self.log_std_head = nn.Linear(layer_size, action_dim)

        # Buffers move to device with .to(device) and are preserved by pickle
        scale = torch.FloatTensor((np.array(action_high) - np.array(action_low)) / 2.0)
        bias = torch.FloatTensor((np.array(action_high) + np.array(action_low)) / 2.0)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)
        # Log-prob correction for the linear scaling is a constant
        self.log_scale_sum = float(torch.log(scale).sum())

    def forward(self, state):
        """Returns (mean, log_std) of the un-squashed Gaussian. Shapes: (batch, action_dim)."""
        x = torch.relu(self.head(state))
        x = torch.relu(self.ff_1(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """Reparameterised sample + tanh squash + scale to action bounds.

        Returns
        -------
        action   : (batch, action_dim) — scaled to [low, high]
        log_prob : (batch,) — log-probability under the policy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()  # reparameterised: enables gradient through sample

        # Tanh squash then scale to action bounds
        a_raw = torch.tanh(u)                    # ∈ (-1, 1)
        action = a_raw * self.scale + self.bias  # ∈ [low, high]

        # Log-prob: Gaussian - tanh Jacobian - linear scaling constant
        # Numerically stable: log(1 - tanh²(u)) = 2*(log2 - u - softplus(-2u))
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        log_prob -= self.log_scale_sum

        return action, log_prob
