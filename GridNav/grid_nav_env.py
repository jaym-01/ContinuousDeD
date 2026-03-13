import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridNavEnv(gym.Env):
    """4x4 continuous gridworld.

    State:  (x, y) ∈ [0, 4]²
    Action: (x_target, y_target) ∈ [0, 4]² — desired next position (direct teleport).

    Zones (evaluated on s_{t+1}):
        Death    (Red):    x ∈ [0, z], y ∈ [4-z, 4]      → r=-1, terminated=True
        Trap     (Yellow): x ∈ [0, z], y ∈ [4-2z, 4-z)   → r=0,  terminated=False
        Recovery (Blue):   x ∈ [3,4],  y ∈ [2,4]         → r=+1, terminated=True
        Neutral  (White):  elsewhere                       → r=0,  terminated=False

    where z = sqrt(dead_end_pct / 2 * SIZE²) is the zone side length derived
    from `dead_end_pct` (fraction in [0,1] of total grid area occupied by death+trap combined).
    Default dead_end_pct=0.125 reproduces the original 1×1 zones.

    Trap mechanism: if s_t is in Trap, the action is ignored and the agent is
    forced to s_{t+1} = (z/2, SIZE-z/2) — the centre of the Death zone.
    """

    SIZE = 4.0
    START = np.array([1.0, 1.0], dtype=np.float32)
    MAX_STEPS = 200

    def __init__(self, dead_end_pct: float = 0.125):
        super().__init__()
        # Each zone (death, trap) occupies half of dead_end_pct of the total grid area.
        zone_area = (dead_end_pct / 2.0) * (self.SIZE ** 2)
        self._z = float(np.sqrt(zone_area))  # side length of each square zone
        self._death_y_lo = self.SIZE - self._z
        self._trap_y_lo = self.SIZE - 2 * self._z
        self._death_center = np.array(
            [self._z / 2, self.SIZE - self._z / 2], dtype=np.float32
        )

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
        return s[0] <= self._z and s[1] >= self._death_y_lo

    def _in_trap(self, s):
        return s[0] <= self._z and self._trap_y_lo <= s[1] < self._death_y_lo

    def _in_recovery(self, s):
        return s[0] >= 3.0 and s[1] >= 2.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.START.copy()
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        if self._in_trap(self.state):
            next_state = self._death_center.copy()
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


class DiscreteGridNavWrapper(gym.Wrapper):
    """Discretises GridNav's 2-D continuous action space into an n×n grid.

    Each discrete action i maps to a (x_target, y_target) pair from an
    n_bins × n_bins uniform grid over [0, 4]².
    Grid is row-major: action i → (bins[i // n_bins], bins[i % n_bins]).
    """

    def __init__(self, env: GridNavEnv, n_bins: int = 5):
        super().__init__(env)
        bins = np.linspace(0.0, GridNavEnv.SIZE, n_bins, dtype=np.float32)
        self._action_map = np.array(
            [(x, y) for x in bins for y in bins], dtype=np.float32
        )  # shape (n_bins², 2)
        self.action_space = gym.spaces.Discrete(len(self._action_map))

    def step(self, action: int):
        target = self._action_map[int(action)]
        return self.env.step(target)


def _make_discrete_gridnav(n_bins: int = 5) -> DiscreteGridNavWrapper:
    return DiscreteGridNavWrapper(GridNavEnv(), n_bins=n_bins)


try:
    gym.register(id="GridNav-discrete-v0", entry_point=_make_discrete_gridnav)
except Exception:
    pass
