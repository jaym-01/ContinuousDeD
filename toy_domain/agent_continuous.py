"""
DSAC: https://arxiv.org/abs/2004.14547
IQN:  https://arxiv.org/abs/1806.06923
"""

import torch
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import random

from model_continuous import ContinuousIQN, ContinuousDQN, GaussianActor
from ReplayBuffers import ReplayBuffer

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def calculate_huber_loss(td_errors, k=1.0):
    return torch.where(
        td_errors.abs() <= k,
        0.5 * td_errors.pow(2),
        k * (td_errors.abs() - 0.5 * k),
    )


# ---------------------------------------------------------------------------
# IQN continuous agent
# ---------------------------------------------------------------------------

class ContinuousIQN_Agent:
    """IQN-based Q(s,a) critic with continuous action selection.

    When use_actor=False (default): action selection via random shooting (K uniform
    candidates evaluated under Q, argmax chosen). Suitable for Q_d / Q_r agents.

    When use_actor=True: DSAC-style Gaussian actor trained with MaxEnt objective
    E[Q(s,a)] + α*H(π). The actor replaces random shooting for both act() and TD
    target construction. Suitable for the main exploration agent.

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
    K_actions    : int — candidate actions for random-shooting mode (use_actor=False)
    worker       : int — number of parallel environments
    device       : str
    seed         : int
    use_actor    : bool — if True, use DSAC Gaussian actor; if False, random shooting
    ALPHA        : float — entropy temperature for DSAC (only used when use_actor=True)
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
        use_actor=False,
        ALPHA=0.2,
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
        self.use_actor = use_actor
        self.alpha = ALPHA

        self.qnetwork_local = ContinuousIQN(
            state_size, action_dim, layer_size, seed, N,
            drm=risk_measure, eta=ETA, device=device,
        ).to(device)
        self.qnetwork_target = ContinuousIQN(
            state_size, action_dim, layer_size, seed, N,
            drm=risk_measure, eta=ETA, device=device,
        ).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        if self.use_actor:
            self.actor_local = GaussianActor(
                state_size, action_dim, layer_size, action_low, action_high, seed,
            ).to(device)
            self.actor_target = GaussianActor(
                state_size, action_dim, layer_size, action_low, action_high, seed,
            ).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
            print(self.qnetwork_local)
            print(self.actor_local)
        else:
            print(self.qnetwork_local)

        self.memory = ReplayBuffer(
            BUFFER_SIZE, self.BATCH_SIZE, device, seed, GAMMA, n_step, worker, continuous=True
        )

    def _action_bounds(self):
        low = torch.FloatTensor(self.action_low).to(self.device)
        high = torch.FloatTensor(self.action_high).to(self.device)
        return low, high

    def act(self, state, eps=0.0, eval=False, use_drm=False):
        """Select action(s) via epsilon-greedy policy.

        With use_actor=True:  greedy branch samples from Gaussian actor.
        With use_actor=False: greedy branch uses random shooting over K candidates.

        Returns
        -------
        numpy array of shape (n, action_dim) where n=1 for eval, n=worker otherwise.
        """
        n = 1 if eval else self.worker

        if random.random() > eps:
            state_t = torch.FloatTensor(np.array(state)).to(self.device)  # (n, state_dim)

            if self.use_actor:
                self.actor_local.eval()
                with torch.no_grad():
                    action, _ = self.actor_local.sample(state_t)  # (n, action_dim)
                self.actor_local.train()
                return action.cpu().numpy()
            else:
                K = self.K_actions
                low, high = self._action_bounds()

                # Expand each state over K candidate actions: (n*K, state_dim)
                state_exp = state_t.unsqueeze(1).expand(n, K, -1).reshape(n * K, -1)
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
            critic_loss, actor_loss = self.learn(self.memory.sample())
            self.Q_updates += 1
            writer.add_scalar("Q_loss", critic_loss, self.Q_updates)
            if actor_loss is not None:
                writer.add_scalar("Actor_loss", actor_loss, self.Q_updates)

    def learn(self, experiences):
        """Distributional TD update with cross-quantile Huber loss.

        With use_actor=True:  DSAC — entropy-augmented Bellman target, actor gradient update.
        With use_actor=False: Standard IQN — random-shooting target, critic update only.

        Returns
        -------
        critic_loss : float
        actor_loss  : float or None (None when use_actor=False)
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # --- Build TD target ---
        with torch.no_grad():
            if self.use_actor:
                # DSAC: next actions from target actor + entropy correction
                next_actions, log_probs = self.actor_target.sample(next_states)
                # next_actions: (batch, action_dim), log_probs: (batch,)
                Q_targets_next, _ = self.qnetwork_target(next_states, next_actions, self.N)
                # Q_targets_next: (batch, N, 1)
                # Subtract entropy: Z(s',a') - α * log π(a'|s')
                Q_targets_next = Q_targets_next - self.alpha * log_probs.view(-1, 1, 1)
            else:
                # Random shooting: sample K candidates, select best, get full distribution
                K = self.K_actions
                low, high = self._action_bounds()
                rand_u = torch.rand(self.BATCH_SIZE * K, self.action_dim, device=self.device)
                rand_actions = rand_u * (high - low) + low  # (batch*K, action_dim)
                ns_exp = (
                    next_states.unsqueeze(1)
                    .expand(self.BATCH_SIZE, K, -1)
                    .reshape(self.BATCH_SIZE * K, -1)
                )  # (batch*K, state_dim)

                q_cand, _ = self.qnetwork_target(ns_exp, rand_actions, self.N)
                # q_cand: (batch*K, N, 1)
                q_cand_mean = q_cand.mean(dim=1).view(self.BATCH_SIZE, K)  # (batch, K)
                best_idx = q_cand_mean.argmax(dim=1)  # (batch,)
                best_actions = rand_actions.view(self.BATCH_SIZE, K, self.action_dim)[
                    torch.arange(self.BATCH_SIZE, device=self.device), best_idx
                ]  # (batch, action_dim)
                Q_targets_next, _ = self.qnetwork_target(next_states, best_actions, self.N)
                # Q_targets_next: (batch, N, 1)

            Q_targets_next = Q_targets_next.detach()

        # DeD clamping (applied after entropy subtraction so the clamping stays meaningful)
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

        # Clamp the final target so n-step accumulation can't push below true bounds
        if self.sided_Q == "negative":
            Q_targets = torch.clamp(Q_targets, max=0.0, min=-1.0)
        elif self.sided_Q == "positive":
            Q_targets = torch.clamp(Q_targets, max=1.0, min=0.0)
        else:
            Q_targets = torch.clamp(Q_targets, max=1.0, min=-1.0)

        # --- Critic loss: cross-quantile Huber (identical structure to discrete IQN) ---
        Q_expected, taus = self.qnetwork_local(states, actions, self.N)
        # Q_expected: (batch, N, 1),  taus: (batch, N, 1)

        # (batch, 1, N) - (batch, N, 1) → (batch, N, N) via broadcasting
        td_error = Q_targets.transpose(1, 2) - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), (
            f"wrong td error shape: {td_error.shape}"
        )
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l
        critic_loss = quantil_l.sum(dim=1).mean(dim=1).mean()

        critic_loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        # --- Actor loss (DSAC only) ---
        if not self.use_actor:
            return critic_loss.detach().cpu().numpy(), None

        self.actor_optimizer.zero_grad()
        actions_new, log_probs_new = self.actor_local.sample(states)
        Q_vals, _ = self.qnetwork_local(states, actions_new, self.N)
        Q_mean = Q_vals.mean(dim=1).squeeze(-1)  # (batch,)
        actor_loss = (self.alpha * log_probs_new - Q_mean).mean()

        actor_loss.backward()
        clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()
        self.soft_update(self.actor_local, self.actor_target)

        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
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

        self.memory = ReplayBuffer(
            BUFFER_SIZE, self.BATCH_SIZE, device, seed, GAMMA, n_step, worker, continuous=True
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
        K = self.K_actions
        low, high = self._action_bounds()

        rand_u = torch.rand(self.BATCH_SIZE * K, self.action_dim, device=self.device)
        rand_actions = rand_u * (high - low) + low
        ns_exp = (
            next_states.unsqueeze(1).expand(self.BATCH_SIZE, K, -1).reshape(self.BATCH_SIZE * K, -1)
        )

        with torch.no_grad():
            Q_cand = self.qnetwork_target(ns_exp, rand_actions)  # (batch*K, 1)
            Q_targets_next = Q_cand.view(self.BATCH_SIZE, K).max(dim=1, keepdim=True).values

        if self.sided_Q == "negative":
            Q_targets_next = torch.clamp(Q_targets_next, max=0.0, min=-1.0)
        elif self.sided_Q == "positive":
            Q_targets_next = torch.clamp(Q_targets_next, max=1.0, min=0.0)
        else:
            Q_targets_next = torch.clamp(Q_targets_next, max=1.0, min=-1.0)

        Q_targets = rewards + self.GAMMA**self.n_step * Q_targets_next * (1.0 - dones)
        Q_expected = self.qnetwork_local(states, actions)  # (batch, 1)

        td_error = Q_targets - Q_expected
        loss = calculate_huber_loss(td_error, 1.0).mean()

        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        for tp, lp in zip(target_model.parameters(), local_model.parameters()):
            tp.data.copy_(self.TAU * lp.data + (1.0 - self.TAU) * tp.data)
