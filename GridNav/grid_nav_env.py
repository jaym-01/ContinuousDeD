import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridNavEnv(gym.Env):
    """4x4 continuous gridworld.

    State:  (x, y) ∈ [0, 4]²
    Action: (x_target, y_target) ∈ [0, 4]² — desired next position (direct teleport).

    Zones (evaluated on s_{t+1}):
        Death    (Red):    x ∈ [0,1], y ∈ [3,4]  → r=-1, terminated=True
        Trap     (Yellow): x ∈ [0,1], y ∈ [2,3)  → r=0,  terminated=False
        Recovery (Blue):   x ∈ [3,4], y ∈ [2,4]  → r=+1, terminated=True
        Neutral  (White):  elsewhere               → r=0,  terminated=False

    Trap mechanism: if s_t is in Trap, the action is ignored and the agent is
    forced to s_{t+1} = (0.5, 3.5) — the centre of the Death zone.
    """

    SIZE = 4.0
    START = np.array([1.0, 1.0], dtype=np.float32)
    MAX_STEPS = 200

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.full(2, self.SIZE, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.full(2, self.SIZE, dtype=np.float32),
            dtype=np.float32,
        )
        self.state = self.START.copy()
        self.steps = 0

    def _in_death(self, s):
        return s[0] <= 1.0 and s[1] >= 3.0

    def _in_trap(self, s):
        return s[0] <= 1.0 and 2.0 <= s[1] < 3.0

    def _in_recovery(self, s):
        return s[0] >= 3.0 and s[1] >= 2.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.START.copy()
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        if self._in_trap(self.state):
            next_state = np.array([0.5, 3.5], dtype=np.float32)
        else:
            next_state = np.clip(
                np.array(action, dtype=np.float32), 0.0, self.SIZE
            )
        self.state = next_state
        self.steps += 1

        if self._in_death(self.state):
            return self.state.copy(), -1.0, True, False, {}
        if self._in_recovery(self.state):
            return self.state.copy(), +1.0, True, False, {}
        truncated = self.steps >= self.MAX_STEPS
        return self.state.copy(), 0.0, False, truncated, {}


try:
    gym.register(id="GridNav-v0", entry_point="grid_nav_env:GridNavEnv")
except Exception:
    pass
