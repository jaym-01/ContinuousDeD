"""
Continuous-action agents for IQN and DQN critics.

Action selection uses uniform random shooting:
  - Sample K candidate actions uniformly from [action_low, action_high]
  - Evaluate Q(s, a_i) for each candidate
  - Return the argmax action (or a random action with probability eps)

TD learning mirrors the discrete IQN/DQN agents:
  - IQN: cross-quantile Huber loss, shape (batch, N, N), same as discrete variant
  - DQN: plain Huber loss on scalar Q values
  - sided_Q clamping for Dead-end Discovery (DeD) is preserved

ContinuousReplayBuffer stores float actions (not integer indices).
"""

import torch
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random
from collections import deque, namedtuple

from model_continuous import ContinuousIQN, ContinuousDQN

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "next_state", "done"]
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _calculate_huber_loss(td_errors, k=1.0):
    return torch.where(
        td_errors.abs() <= k,
        0.5 * td_errors.pow(2),
        k * (td_errors.abs() - 0.5 * k),
    )


class ContinuousReplayBuffer:
    """Replay buffer that stores continuous (float) actions.

    Identical to ReplayBuffer in ReplayBuffers.py except sample() returns
    actions as FloatTensor instead of LongTensor.
    """

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1, parallel_env=1):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.parallel_env = parallel_env
        self.n_step_buffer = [deque(maxlen=self.n_step) for _ in range(parallel_env)]
        self.iter_ = 0

    def add(self, state, action, reward, next_state, done):
        if self.iter_ == self.parallel_env:
            self.iter_ = 0
        self.n_step_buffer[self.iter_].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[self.iter_]) == self.n_step:
            state, action, reward, next_state, done = self._calc_multistep_return(
                self.n_step_buffer[self.iter_]
            )
            self.memory.append(self.experience(state, action, reward, next_state, done))
        self.iter_ += 1

    def _calc_multistep_return(self, n_step_buffer):
        ret = sum(self.gamma**i * n_step_buffer[i][2] for i in range(self.n_step))
        return n_step_buffer[0][0], n_step_buffer[0][1], ret, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.stack([e.state for e in experiences])
        ).float().to(self.device)
        # Float actions (shape: batch x action_dim)
        actions = torch.from_numpy(
            np.stack([e.action for e in experiences])
        ).float().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences])
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences])
        ).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences]).astype(np.uint8)
        ).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# IQN continuous agent
# ---------------------------------------------------------------------------

class ContinuousIQN_Agent:
    """IQN-based Q(s,a) critic with continuous action selection via random shooting.

    Parameters
    ----------
    state_size   : tuple, e.g. (4,)
    action_dim   : int — dimensionality of continuous action space
    action_low   : array-like — per-dimension lower bounds
    action_high  : array-like — per-dimension upper bounds
    network      : str — passed for noisy/dueling flags (same as discrete agent)
    munchausen   : int — unused for continuous (no softmax over actions)
    layer_size   : int
    n_step       : int
    risk_measure : str — DRM for tau sampling: identity|cvar|cpw|power
    sided_Q      : str — 'both'|'negative'|'positive' for DeD clamping
    ETA          : float — DRM scaling factor
    BATCH_SIZE   : int
    BUFFER_SIZE  : int
    LR           : float
    TAU          : float — Polyak averaging coefficient
    GAMMA        : float
    N            : int — quantile samples during training
    K_actions    : int — candidate actions sampled for action selection and TD targets
    worker       : int — number of parallel environments
    device       : str
    seed         : int
    """

    def __init__(
        self,
        state_size,
        action_dim,
        action_low,
        action_high,
        network,
        munchausen,
        layer_size,
        n_step,
        risk_measure,
        sided_Q,
        ETA,
        BATCH_SIZE,
        BUFFER_SIZE,
        LR,
        TAU,
        GAMMA,
        N,
        K_actions,
        worker,
        device,
        seed,
    ):
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_low = np.array(action_low, dtype=np.float32)
        self.action_high = np.array(action_high, dtype=np.float32)
        self.network = network
        self.seed = random.seed(seed)
        self.seed_t = torch.manual_seed(seed)
        self.device = device
        self.sided_Q = sided_Q
        self.TAU = TAU
        self.N = N
        self.K_actions = K_actions
        self.GAMMA = GAMMA
        self.n_step = n_step
        self.worker = worker
        self.BATCH_SIZE = BATCH_SIZE * worker
        self.UPDATE_EVERY = worker
        self.Q_updates = 0
        self.t_step = 0

        self.qnetwork_local = ContinuousIQN(
            state_size, action_dim, layer_size, seed, N,
            drm=risk_measure, eta=ETA, device=device,
        ).to(device)
        self.qnetwork_target = ContinuousIQN(
            state_size, action_dim, layer_size, seed, N,
            drm=risk_measure, eta=ETA, device=device,
        ).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)

        self.memory = ContinuousReplayBuffer(
            BUFFER_SIZE, self.BATCH_SIZE, device, seed, GAMMA, n_step, worker
        )

    def _action_bounds(self):
        low = torch.FloatTensor(self.action_low).to(self.device)
        high = torch.FloatTensor(self.action_high).to(self.device)
        return low, high

    def act(self, state, eps=0.0, eval=False, use_drm=False):
        """Select action(s) via epsilon-greedy random shooting.

        Returns
        -------
        numpy array of shape (n, action_dim) where n=1 for eval, n=worker otherwise.
        """
        n = 1 if eval else self.worker

        if random.random() > eps:
            state_t = torch.FloatTensor(np.array(state)).to(self.device)  # (n, state_dim)
            K = self.K_actions
            low, high = self._action_bounds()

            # Expand each state over K candidate actions: (n*K, state_dim)
            state_exp = state_t.unsqueeze(1).expand(n, K, -1).reshape(n * K, -1)

            # Sample K uniform random actions per state: (n*K, action_dim)
            rand_u = torch.rand(n * K, self.action_dim, device=self.device)
            rand_actions = rand_u * (high - low) + low

            self.qnetwork_local.eval()
            with torch.no_grad():
                q_vals = self.qnetwork_local.get_qvalue(
                    state_exp, rand_actions, use_drm
                )  # (n*K, 1)
            self.qnetwork_local.train()

            q_vals = q_vals.view(n, K)  # (n, K)
            best_idx = q_vals.argmax(dim=1)  # (n,)
            rand_actions = rand_actions.view(n, K, self.action_dim)
            best_actions = rand_actions[
                torch.arange(n, device=self.device), best_idx
            ]  # (n, action_dim)
            return best_actions.cpu().numpy()
        else:
            low, high = self.action_low, self.action_high
            return np.stack([
                np.random.uniform(low, high).astype(np.float32) for _ in range(n)
            ])

    def step(self, state, action, reward, next_state, done, writer):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.BATCH_SIZE:
            loss = self.learn(self.memory.sample())
            self.Q_updates += 1
            writer.add_scalar("Q_loss", loss, self.Q_updates)

    def learn(self, experiences):
        """Distributional TD update with cross-quantile Huber loss (same (batch,N,N)
        structure as the discrete IQN agent)."""
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        batch = states.shape[0]
        K = self.K_actions
        low, high = self._action_bounds()

        # --- Build TD target ---
        # Sample K candidate actions for each next_state
        rand_u = torch.rand(batch * K, self.action_dim, device=self.device)
        rand_actions = rand_u * (high - low) + low  # (batch*K, action_dim)
        ns_exp = (
            next_states.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        )  # (batch*K, state_dim)

        with torch.no_grad():
            # Mean Q over N quantile samples → select best action per next_state
            q_cand, _ = self.qnetwork_target(ns_exp, rand_actions, self.N)
            # q_cand: (batch*K, N, 1)
            q_cand_mean = q_cand.mean(dim=1).view(batch, K)  # (batch, K)
            best_idx = q_cand_mean.argmax(dim=1)  # (batch,)

            best_actions = rand_actions.view(batch, K, self.action_dim)[
                torch.arange(batch, device=self.device), best_idx
            ]  # (batch, action_dim)

            # Full quantile distribution at best action (N target samples)
            Q_targets_next, _ = self.qnetwork_target(
                next_states, best_actions, self.N
            )  # (batch, N, 1)
            Q_targets_next = Q_targets_next.detach()

        # DeD clamping
        if self.sided_Q == "negative":
            Q_targets_next = torch.clamp(Q_targets_next, max=0.0, min=-1.0)
        elif self.sided_Q == "positive":
            Q_targets_next = torch.clamp(Q_targets_next, max=1.0, min=0.0)
        else:
            Q_targets_next = torch.clamp(Q_targets_next, max=1.0, min=-1.0)

        # Bellman backup: (batch, N, 1)
        Q_targets = rewards.unsqueeze(-1) + (
            self.GAMMA**self.n_step
            * Q_targets_next
            * (1.0 - dones.unsqueeze(-1))
        )

        # --- Expected Q from local network ---
        Q_expected, taus = self.qnetwork_local(states, actions, self.N)
        # Q_expected: (batch, N, 1),  taus: (batch, N, 1)

        # Cross-quantile Huber loss (identical structure to discrete IQN)
        # (batch, 1, N) - (batch, N, 1) → (batch, N, N) via broadcasting
        td_error = Q_targets.transpose(1, 2) - Q_expected
        assert td_error.shape == (batch, self.N, self.N), (
            f"wrong td error shape: {td_error.shape}"
        )
        huber_l = _calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l
        loss = quantil_l.sum(dim=1).mean(dim=1).mean()

        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def _soft_update(self, local, target):
        for tp, lp in zip(target.parameters(), local.parameters()):
            tp.data.copy_(self.TAU * lp.data + (1.0 - self.TAU) * tp.data)


# ---------------------------------------------------------------------------
# DQN continuous agent
# ---------------------------------------------------------------------------

class ContinuousDQN_Agent:
    """DQN-based Q(s,a) critic with continuous action selection via random shooting.

    Parameters match ContinuousIQN_Agent; risk_measure, ETA, N are unused/ignored.
    """

    def __init__(
        self,
        state_size,
        action_dim,
        action_low,
        action_high,
        network,
        munchausen,
        layer_size,
        n_step,
        sided_Q,
        BATCH_SIZE,
        BUFFER_SIZE,
        LR,
        TAU,
        GAMMA,
        K_actions,
        worker,
        device,
        seed,
        **kwargs,  # absorb unused IQN-specific kwargs (risk_measure, ETA, N)
    ):
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_low = np.array(action_low, dtype=np.float32)
        self.action_high = np.array(action_high, dtype=np.float32)
        self.network = network
        self.seed = random.seed(seed)
        self.seed_t = torch.manual_seed(seed)
        self.device = device
        self.sided_Q = sided_Q
        self.TAU = TAU
        self.K_actions = K_actions
        self.GAMMA = GAMMA
        self.n_step = n_step
        self.worker = worker
        self.BATCH_SIZE = BATCH_SIZE * worker
        self.UPDATE_EVERY = worker
        self.Q_updates = 0
        self.t_step = 0

        self.qnetwork_local = ContinuousDQN(
            state_size, action_dim, layer_size, seed, device=device
        ).to(device)
        self.qnetwork_target = ContinuousDQN(
            state_size, action_dim, layer_size, seed, device=device
        ).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)

        self.memory = ContinuousReplayBuffer(
            BUFFER_SIZE, self.BATCH_SIZE, device, seed, GAMMA, n_step, worker
        )

    def _action_bounds(self):
        low = torch.FloatTensor(self.action_low).to(self.device)
        high = torch.FloatTensor(self.action_high).to(self.device)
        return low, high

    def act(self, state, eps=0.0, eval=False, **kwargs):
        """Select action(s) via epsilon-greedy random shooting. Returns (n, action_dim)."""
        n = 1 if eval else self.worker

        if random.random() > eps:
            state_t = torch.FloatTensor(np.array(state)).to(self.device)
            K = self.K_actions
            low, high = self._action_bounds()

            state_exp = state_t.unsqueeze(1).expand(n, K, -1).reshape(n * K, -1)
            rand_u = torch.rand(n * K, self.action_dim, device=self.device)
            rand_actions = rand_u * (high - low) + low

            self.qnetwork_local.eval()
            with torch.no_grad():
                q_vals = self.qnetwork_local.get_qvalue(state_exp, rand_actions)  # (n*K, 1)
            self.qnetwork_local.train()

            q_vals = q_vals.view(n, K)
            best_idx = q_vals.argmax(dim=1)
            rand_actions = rand_actions.view(n, K, self.action_dim)
            best_actions = rand_actions[
                torch.arange(n, device=self.device), best_idx
            ]
            return best_actions.cpu().numpy()
        else:
            low, high = self.action_low, self.action_high
            return np.stack([
                np.random.uniform(low, high).astype(np.float32) for _ in range(n)
            ])

    def step(self, state, action, reward, next_state, done, writer):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > self.BATCH_SIZE:
            loss = self.learn(self.memory.sample())
            self.Q_updates += 1
            writer.add_scalar("Q_loss", loss, self.Q_updates)

    def learn(self, experiences):
        """Plain Huber loss on scalar Q(s,a) values."""
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        batch = states.shape[0]
        K = self.K_actions
        low, high = self._action_bounds()

        rand_u = torch.rand(batch * K, self.action_dim, device=self.device)
        rand_actions = rand_u * (high - low) + low
        ns_exp = (
            next_states.unsqueeze(1).expand(batch, K, -1).reshape(batch * K, -1)
        )

        with torch.no_grad():
            Q_cand = self.qnetwork_target(ns_exp, rand_actions)  # (batch*K, 1)
            Q_targets_next = Q_cand.view(batch, K).max(dim=1, keepdim=True).values

        if self.sided_Q == "negative":
            Q_targets_next = torch.clamp(Q_targets_next, max=0.0, min=-1.0)
        elif self.sided_Q == "positive":
            Q_targets_next = torch.clamp(Q_targets_next, max=1.0, min=0.0)
        else:
            Q_targets_next = torch.clamp(Q_targets_next, max=1.0, min=-1.0)

        Q_targets = rewards + self.GAMMA**self.n_step * Q_targets_next * (1.0 - dones)
        Q_expected = self.qnetwork_local(states, actions)  # (batch, 1)

        td_error = Q_targets - Q_expected
        loss = _calculate_huber_loss(td_error, 1.0).mean()

        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def _soft_update(self, local, target):
        for tp, lp in zip(target.parameters(), local.parameters()):
            tp.data.copy_(self.TAU * lp.data + (1.0 - self.TAU) * tp.data)
