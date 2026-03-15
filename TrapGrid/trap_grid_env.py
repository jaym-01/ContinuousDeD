import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TrapGridEnv(gym.Env):
    """10x10 continuous gridworld with a circular trap/recovery zone and red death zone.

    State:  (x, y) ∈ [0, 10]²
    Action: (x_target, y_target) ∈ [0, 10]² — desired next position (direct teleport).
    Start:  (9.9, 0.1) — bottom-right corner.

    Zones (evaluated on s_{t+1}):
        Recovery Zone (Blue):  (x-5)²+(y-5)² ≤ 0.0625  (r=0.25)  → r=+1, terminated=True
        Trap Zone     (Yellow): 0.0625 < (x-5)²+(y-5)² ≤ 1.0       → r=0,  terminated=False
        Neutral Zone  (White): x ∈ [9.8,10], y ∈ [0,0.2]           → r=0,  terminated=False
        Death Zone    (Red):   everywhere else                        → r=-1, terminated=True

    Trap mechanics:
        If s_t is in Trap Zone, action is nullified and s_{t+1} = (2.0, 2.0) (Death Zone).
    """

    SIZE = 10.0
    START = np.array([9.9, 0.1], dtype=np.float32)
    MAX_STEPS = 200

    _RECOVERY_CENTER = np.array([5.0, 5.0], dtype=np.float32)
    _RECOVERY_RADIUS = 0.25
    _TRAP_OUTER_RADIUS = 1.0
    _TRAP_FORCED_STATE = np.array([2.0, 2.0], dtype=np.float32)

    # Neutral (start) zone bounds
    _NEUTRAL_X_MIN = 9.8
    _NEUTRAL_X_MAX = 10.0
    _NEUTRAL_Y_MIN = 0.0
    _NEUTRAL_Y_MAX = 0.2

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self._recovery_radius = self._RECOVERY_RADIUS * scale
        self._trap_outer_radius = self._TRAP_OUTER_RADIUS * scale
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

    def _dist_center(self, s):
        return float(np.sqrt(
            (s[0] - self._RECOVERY_CENTER[0]) ** 2 +
            (s[1] - self._RECOVERY_CENTER[1]) ** 2
        ))

    def _in_recovery(self, s):
        return self._dist_center(s) <= self._recovery_radius

    def _in_trap(self, s):
        d = self._dist_center(s)
        return self._recovery_radius < d <= self._trap_outer_radius

    def _in_neutral(self, s):
        return (
            self._NEUTRAL_X_MIN <= s[0] <= self._NEUTRAL_X_MAX and
            self._NEUTRAL_Y_MIN <= s[1] <= self._NEUTRAL_Y_MAX
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
        # Trap mechanics: if currently in trap zone, force into death zone
        if self._in_trap(self.state):
            next_state = self._TRAP_FORCED_STATE.copy()
        else:
            next_state = np.clip(
                np.array(action, dtype=np.float32), 0.0, self.SIZE
            )
        self.state = next_state
        self.steps += 1

        if self._in_recovery(self.state):
            return self.state.copy(), +1.0, True, False, {}
        if not (self._in_neutral(self.state) or self._in_trap(self.state)):
            # Death zone is the complement of recovery, trap, and neutral
            return self.state.copy(), -1.0, True, False, {}
        truncated = self.steps >= self.MAX_STEPS
        return self.state.copy(), 0.0, False, truncated, {}


try:
    gym.register(id="TrapGrid-v0", entry_point="trap_grid_env:TrapGridEnv")
except Exception:
    pass


class DiscreteTrapGridWrapper(gym.Wrapper):
    """Discretises TrapGrid's 2-D continuous action space into an n×n grid.

    Each discrete action i maps to a (x_target, y_target) pair from an
    n_bins × n_bins uniform grid over [0, 10]².
    Grid is row-major: action i → (bins[i // n_bins], bins[i % n_bins]).
    """

    def __init__(self, env: TrapGridEnv, n_bins: int = 5):
        super().__init__(env)
        bins = np.linspace(0.0, TrapGridEnv.SIZE, n_bins, dtype=np.float32)
        self._action_map = np.array(
            [(x, y) for x in bins for y in bins], dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(self._action_map))

    def step(self, action: int):
        target = self._action_map[int(action)]
        return self.env.step(target)


def _make_discrete_trapgrid(n_bins: int = 5, scale: float = 1.0) -> DiscreteTrapGridWrapper:
    return DiscreteTrapGridWrapper(TrapGridEnv(scale=scale), n_bins=n_bins)


try:
    gym.register(id="TrapGrid-discrete-v0", entry_point=_make_discrete_trapgrid)
except Exception:
    pass
