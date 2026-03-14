import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MedGridHardEnv(gym.Env):
    """10x10 continuous gridworld with an L-shaped danger zone and a circular recovery zone.

    State:  (x, y) ∈ [0, 10]²
    Action: (x_target, y_target) ∈ [0, 10]² — desired next position (direct teleport).
    Start:  (10, 0) — bottom-right corner.

    Zones (evaluated on s_{t+1}):
        Death Zone 1  (Red):    dist_death1(x,y) ≤ 2                            → r=-1, terminated=True
        Trap Zone 1   (Yellow): 2 < dist_death1(x,y) ≤ 4                        → r=0,  terminated=False
        Death Zone 2  (Red):    rotated ellipse at (9,5), a=1, b=1.5, θ=+45°    → r=-1, terminated=True
        Trap Zone 2   (Yellow): rotated ellipse at (9,5), a=1.5, b=2.5, θ=+45°  → r=0,  terminated=False
        Death Zone 3  (Red):    rotated ellipse at (5.5,5.5), a=2, b=1, θ=-45°      → r=-1, terminated=True
        Trap Zone 3   (Yellow): rotated ellipse at (5.5,5.5), a=2.5, b=1.5, θ=-45°  → r=0,  terminated=False
        Recovery Zone (Blue):   dist_recovery(x,y) ≤ 1                            → r=+1, terminated=True
        Neutral       (White):  elsewhere                                       → r=0,  terminated=False

    where dist_death1 = sqrt(x²+y²)  and  dist_recovery = sqrt((x-7.5)²+(y-8)²).

    Rotated ellipse condition for zone 2:
        ((x-9)-(y-9))² / (2a²) + ((x-5)+(y-5))² / (2b²) ≤ 1

    Rotated ellipse condition for zone 3:
    ((x-5.5)-(y-5.5))² / (2a²) + ((x-5.5)+(y-5.5))² / (2b²) ≤ 1

    Trap mechanics:
        Trap 1 → forced next state (1, 1)  (inside Death Zone 1)
        Trap 2 → forced next state (9, 5)  (inside Death Zone 2)
        Trap 2 → forced next state (5.5, 5.5)  (inside Death Zone 3)
    """

    SIZE = 10.0
    START = np.array([10.0, 0.0], dtype=np.float32)
    MAX_STEPS = 200

    _DEATH1_RADIUS = 3.0
    _TRAP1_RADIUS = 4.0
    _DEATH2_ELLIPSE_A = 1.0
    _DEATH2_ELLIPSE_B = 1.5
    _TRAP2_ELLIPSE_A = 1.5
    _TRAP2_ELLIPSE_B = 2.5
    _DEATH3_ELLIPSE_A = 2.0
    _DEATH3_ELLIPSE_B = 1.0
    _TRAP3_ELLIPSE_A = 3
    _TRAP3_ELLIPSE_B = 2
    _RECOVERY_RADIUS = 0.5
    _RECOVERY_CENTER = np.array([7.75, 6.75], dtype=np.float32)
    _DEATH1_CENTER = np.array([1, 1], dtype=np.float32)
    _DEATH2_CENTER = np.array([9, 5], dtype=np.float32)
    _DEATH3_CENTER = np.array([5, 6], dtype=np.float32)
    _TRAP1_CENTER = np.array([1, 1], dtype=np.float32)
    _TRAP2_CENTER = np.array([9, 5], dtype=np.float32)
    _TRAP3_CENTER = np.array([5, 6], dtype=np.float32)

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self._DEATH1_RADIUS = MedGridHardEnv._DEATH1_RADIUS * scale
        self._TRAP1_RADIUS = MedGridHardEnv._TRAP1_RADIUS * scale
        self._DEATH2_ELLIPSE_A = MedGridHardEnv._DEATH2_ELLIPSE_A * scale
        self._DEATH2_ELLIPSE_B = MedGridHardEnv._DEATH2_ELLIPSE_B * scale
        self._TRAP2_ELLIPSE_A = MedGridHardEnv._TRAP2_ELLIPSE_A * scale
        self._TRAP2_ELLIPSE_B = MedGridHardEnv._TRAP2_ELLIPSE_B * scale
        self._DEATH3_ELLIPSE_A = MedGridHardEnv._DEATH3_ELLIPSE_A * scale
        self._DEATH3_ELLIPSE_B = MedGridHardEnv._DEATH3_ELLIPSE_B * scale
        self._TRAP3_ELLIPSE_A = MedGridHardEnv._TRAP3_ELLIPSE_A * scale
        self._TRAP3_ELLIPSE_B = MedGridHardEnv._TRAP3_ELLIPSE_B * scale
        self._RECOVERY_RADIUS = MedGridHardEnv._RECOVERY_RADIUS * scale
        self._DEATH1_CENTER = MedGridHardEnv._DEATH1_CENTER * scale
        self._DEATH2_CENTER = MedGridHardEnv._DEATH2_CENTER * scale
        self._DEATH3_CENTER = MedGridHardEnv._DEATH3_CENTER * scale
        self._TRAP1_CENTER = MedGridHardEnv._TRAP1_CENTER * scale
        self._TRAP2_CENTER = MedGridHardEnv._TRAP2_CENTER * scale
        self._TRAP3_CENTER = MedGridHardEnv._TRAP3_CENTER * scale
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
    
    def _dist_death1(self, s):
        return float(np.sqrt(s[0] ** 2 + s[1] ** 2))
    
    def _dist_recovery(self, s):
        return float(np.sqrt((s[0] - self._RECOVERY_CENTER[0]) ** 2 + (s[1] - self._RECOVERY_CENTER[1]) ** 2))

    def _in_recovery(self, s):
        d = self._dist_recovery(s)
        return d <= self._RECOVERY_RADIUS

    def _in_death1(self, s):
        d = self._dist_death1(s)
        return d <= self._DEATH1_RADIUS

    def _in_death2(self, s):
        dx = s[0] - self._DEATH2_CENTER[0]
        dy = s[1] - self._DEATH2_CENTER[1]
        a, b = self._DEATH2_ELLIPSE_A, self._DEATH2_ELLIPSE_B
        return (
            ((dx - dy) ** 2) / (2 * a ** 2) + ((dx + dy) ** 2) / (2 * b ** 2) <= 1.0
        )

    def _in_death3(self, s):
        dx = s[0] - self._DEATH3_CENTER[0]
        dy = s[1] - self._DEATH3_CENTER[1]
        a, b = self._DEATH3_ELLIPSE_A, self._DEATH3_ELLIPSE_B
        return (
            ((dx - dy) ** 2) / (2 * a ** 2) + ((dx + dy) ** 2) / (2 * b ** 2) <= 1.0
        )

    def _in_trap1(self, s):
        d = self._dist_death1(s)
        return self._DEATH1_RADIUS < d <= self._TRAP1_RADIUS

    def _in_trap2(self, s):
        dx = s[0] - self._TRAP2_CENTER[0]
        dy = s[1] - self._TRAP2_CENTER[1]
        a, b = self._TRAP2_ELLIPSE_A, self._TRAP2_ELLIPSE_B
        return (
            ((dx - dy) ** 2) / (2 * a ** 2) + ((dx + dy) ** 2) / (2 * b ** 2) <= 1.0
        )
    
    def _in_trap3(self, s):
        dx = s[0] - self._TRAP3_CENTER[0]
        dy = s[1] - self._TRAP3_CENTER[1]
        a, b = self._TRAP3_ELLIPSE_A, self._TRAP3_ELLIPSE_B
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
        elif self._in_trap3(self.state):
            next_state = self._TRAP3_CENTER.copy()
        else:
            next_state = np.clip(
                np.array(action, dtype=np.float32), 0.0, self.SIZE
            )
        self.state = next_state
        self.steps += 1

        if self._in_death1(self.state) or self._in_death2(self.state) or self._in_death3(self.state):
            return self.state.copy(), -1.0, True, False, {}
        if self._in_recovery(self.state):
            return self.state.copy(), +1.0, True, False, {}
        truncated = self.steps >= self.MAX_STEPS
        return self.state.copy(), 0.0, False, truncated, {}


try:
    gym.register(id="MedGridHard-v0", entry_point="med_grid_hard_env:MedGridHardEnv")
except Exception:
    pass


class DiscreteMedGridHardWrapper(gym.Wrapper):
    """Discretises MedGridHard's 2-D continuous action space into an n×n grid.

    Each discrete action i maps to a (x_target, y_target) pair from an
    n_bins × n_bins uniform grid over [0, 10]².
    Grid is row-major: action i → (bins[i // n_bins], bins[i % n_bins]).
    """

    def __init__(self, env: MedGridHardEnv, n_bins: int = 5):
        super().__init__(env)
        bins = np.linspace(0.0, MedGridHardEnv.SIZE, n_bins, dtype=np.float32)
        self._action_map = np.array(
            [(x, y) for x in bins for y in bins], dtype=np.float32
        )  # shape (n_bins², 2)
        self.action_space = gym.spaces.Discrete(len(self._action_map))

    def step(self, action: int):
        target = self._action_map[int(action)]
        return self.env.step(target)


def _make_discrete_medgridhard(n_bins: int = 5, scale: float = 1.0) -> DiscreteMedGridHardWrapper:
    return DiscreteMedGridHardWrapper(MedGridHardEnv(scale=scale), n_bins=n_bins)


try:
    gym.register(id="MedGridHard-discrete-v0", entry_point=_make_discrete_medgridhard)
except Exception:
    pass
