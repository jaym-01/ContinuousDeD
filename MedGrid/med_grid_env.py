import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MedGridEnv(gym.Env):
    """10x10 continuous gridworld with radial danger zones and a tilted elliptical recovery zone.

    State:  (x, y) ∈ [0, 10]²
    Action: (x_target, y_target) ∈ [0, 10]² — desired next position (direct teleport).
    Start:  (10, 0) — bottom-right corner.

    Zones (evaluated on s_{t+1}):
        Death Zone 1  (Red):    d_BL(x,y) ≤ 3                     → r=-1, terminated=True
        Trap Zone 1   (Yellow): 3 < d_BL(x,y) ≤ 5                 → r=0,  terminated=False
        Death Zone 2  (Red):    d_TR(x,y) ≤ 3                     → r=-1, terminated=True
        Trap Zone 2   (Yellow): 3 < d_TR(x,y) ≤ 5                 → r=0,  terminated=False
        Recovery Zone (Blue):   rotated ellipse at (5,5), a=3, b=1.5, θ=-45° → r=+1, terminated=True
        Neutral       (White):  elsewhere                           → r=0,  terminated=False

    where d_BL = sqrt(x²+y²)  and  d_TR = sqrt((x-10)²+(y-10)²).

    Rotated ellipse condition:
        ((x-5)-(y-5))² / (2a²) + ((x-5)+(y-5))² / (2b²) ≤ 1

    Trap mechanics:
        Trap 1 → forced next state (1.5, 1.5)  (inside Death Zone 1)
        Trap 2 → forced next state (8.5, 8.5)  (inside Death Zone 2)
    """

    SIZE = 10.0
    START = np.array([10.0, 0.0], dtype=np.float32)
    MAX_STEPS = 200

    _DEATH_RADIUS = 3.0
    _TRAP_RADIUS = 5.0
    _ELLIPSE_A = 3.0
    _ELLIPSE_B = 1.5
    _TRAP1_CENTER = np.array([1.5, 1.5], dtype=np.float32)
    _TRAP2_CENTER = np.array([8.5, 8.5], dtype=np.float32)

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

    def _d_bl(self, s):
        return float(np.sqrt(s[0] ** 2 + s[1] ** 2))

    def _d_tr(self, s):
        return float(np.sqrt((s[0] - 10.0) ** 2 + (s[1] - 10.0) ** 2))

    def _in_death(self, s):
        return self._d_bl(s) <= self._DEATH_RADIUS or self._d_tr(s) <= self._DEATH_RADIUS

    def _in_trap1(self, s):
        d = self._d_bl(s)
        return self._DEATH_RADIUS < d <= self._TRAP_RADIUS

    def _in_trap2(self, s):
        d = self._d_tr(s)
        return self._DEATH_RADIUS < d <= self._TRAP_RADIUS

    def _in_recovery(self, s):
        dx = s[0] - 5.0
        dy = s[1] - 5.0
        a, b = self._ELLIPSE_A, self._ELLIPSE_B
        return (
            ((dx - dy) ** 2) / (2 * a ** 2) + ((dx + dy) ** 2) / (2 * b ** 2) <= 1.0
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "start_state" in options:
            self.state = np.array(options["start_state"], dtype=np.float32)
        else:
            self.state = self.START.copy()
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        # Trap mechanics: override action if currently trapped
        if self._in_trap1(self.state):
            next_state = self._TRAP1_CENTER.copy()
        elif self._in_trap2(self.state):
            next_state = self._TRAP2_CENTER.copy()
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
    gym.register(id="MedGrid-v0", entry_point="med_grid_env:MedGridEnv")
except Exception:
    pass


class DiscreteMedGridWrapper(gym.Wrapper):
    """Discretises MedGrid's 2-D continuous action space into an n×n grid.

    Each discrete action i maps to a (x_target, y_target) pair from an
    n_bins × n_bins uniform grid over [0, 10]².
    Grid is row-major: action i → (bins[i // n_bins], bins[i % n_bins]).
    """

    def __init__(self, env: MedGridEnv, n_bins: int = 5):
        super().__init__(env)
        bins = np.linspace(0.0, MedGridEnv.SIZE, n_bins, dtype=np.float32)
        self._action_map = np.array(
            [(x, y) for x in bins for y in bins], dtype=np.float32
        )  # shape (n_bins², 2)
        self.action_space = gym.spaces.Discrete(len(self._action_map))

    def step(self, action: int):
        target = self._action_map[int(action)]
        return self.env.step(target)


def _make_discrete_medgrid(n_bins: int = 5) -> DiscreteMedGridWrapper:
    return DiscreteMedGridWrapper(MedGridEnv(), n_bins=n_bins)


try:
    gym.register(id="MedGrid-discrete-v0", entry_point=_make_discrete_medgrid)
except Exception:
    pass
