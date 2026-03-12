"""
SpaceEnv: A continuous-control navigation environment for dead-end research.

A spaceship navigates a 10×10 map from near (0.5, 0.5) to a target at (9.5, 9.5),
avoiding a gravitational planet at (5, 5). Near the critical radius r_c = sqrt(GM/a_max),
gravity exceeds maximum thrust and escape becomes impossible — a true dead end.

Exports
-------
SpaceEnv          : gymnasium.Env subclass (registered as "SpaceEnv-v0")
R_C               : float — analytical critical radius (2.0 for default params)
flatten_obs       : observation dict → flat [x, y, vx, vy] vector
get_action        : policy-name → 2-D thrust action
collect_dataset   : run mixed-policy episodes and return a labelled DataFrame
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_C = 2.0                              # Critical radius for default planet params (GM=2, a_max=0.5)
PLANET_CENTER = np.array([5.0, 5.0])  # Planet position (centre of a 10×10 map)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SpaceEnv(gym.Env):
    """Spaceship navigation environment with gravitational dead ends.

    Parameters
    ----------
    size : float
        Side length of the square map.
    planet_specs : list[list[float]]
        Each inner list is [radius, GM] for one planet.  All planets are
        placed at the map centre (size/2, size/2).
    max_acceleration : float
        Maximum thrust magnitude per axis (m/s²).
    timestep : float
        Duration of each simulation step (s).
    sigma_noise : float
        Standard deviation of Gaussian noise added to each action component.
    """

    def __init__(
        self,
        size: float = 10,
        planet_specs: list = [[1, 2]],
        max_acceleration: float = 0.5,
        timestep: float = 0.1,
        sigma_noise: float = 0.05,
    ):
        self.size = size
        self.num_planets = len(planet_specs)
        self.planet_specs = np.array(planet_specs, dtype=np.float32)
        self.max_acceleration = max_acceleration
        self.timestep = timestep
        self.target_radius = 0.5
        self.sigma_noise = sigma_noise

        self._agent_position = np.zeros(2, dtype=np.float32)
        self._agent_velocity = np.zeros(2, dtype=np.float32)
        self._target_position = np.array([9.5, 9.5], dtype=np.float32)
        self._planet_positions = np.full(
            (self.num_planets, 2), fill_value=size / 2, dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict({
            "agent position": gym.spaces.Box(0, size, shape=(2,), dtype=np.float32),
            "agent velocity": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            ),
            "target": gym.spaces.Box(0, size, shape=(2,), dtype=np.float32),
            "planets": gym.spaces.Box(0, size, shape=(self.num_planets, 2), dtype=np.float32),
        })

        self.action_space = gym.spaces.Box(
            low=-self.max_acceleration,
            high=self.max_acceleration,
            shape=(2,),
            dtype=np.float32,
        )

        self.step_counter = 0
        self.distance_to_planets = np.zeros(self.num_planets, dtype=np.float32)

    def _get_obs(self) -> dict:
        return {
            "agent position": self._agent_position.copy(),
            "agent velocity": self._agent_velocity.copy(),
            "target": self._target_position.copy(),
            "planets": self._planet_positions.copy(),
        }

    def _get_info(self) -> dict:
        return {
            "distance to target": np.linalg.norm(
                self._agent_position - self._target_position
            ),
            "distance to planets": self.distance_to_planets,
            "planet radiuses and G*M coef": self.planet_specs,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.step_counter = 0

        # If a start_state is provided, use it
        if options is not None and "start_state" in options:
            s = options["start_state"]
            self._agent_position = np.array([s[0], s[1]], dtype=np.float32)
            self._agent_velocity = np.array([s[2], s[3]], dtype=np.float32)
        else:

            start_x = self.np_random.uniform(0.1, 1.0)
            start_y = self.np_random.uniform(0.1, 1.0)
            self._agent_position = np.array([start_x, start_y], dtype=np.float32)
            self._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)

        self._target_position = np.array([9.5, 9.5], dtype=np.float32)
        self._planet_positions = np.full(
            (self.num_planets, 2), fill_value=self.size / 2, dtype=np.float32
        )

        self.distance_to_planets = np.array(
            [np.linalg.norm(self._agent_position - p) for p in self._planet_positions],
            dtype=np.float32,
        )

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(
            np.array(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        self.distance_to_planets = np.array(
            [np.linalg.norm(self._agent_position - p) for p in self._planet_positions],
            dtype=np.float32,
        )

        gravity_acceleration = np.zeros(2, dtype=np.float32)
        for i in range(self.num_planets):
            dist = max(self.distance_to_planets[i], 1e-5)
            relative_position = self._agent_position - self._planet_positions[i]
            gravity_acceleration += -self.planet_specs[i][1] * relative_position / (dist ** 3)

        noise = self.np_random.normal(scale=self.sigma_noise, size=2).astype(np.float32)
        total_acceleration = action + gravity_acceleration + noise
        self._agent_velocity += total_acceleration * self.timestep
        self._agent_position += self._agent_velocity * self.timestep

        distance_to_target = np.linalg.norm(self._agent_position - self._target_position)
        success = distance_to_target <= self.target_radius

        self.distance_to_planets = np.array(
            [np.linalg.norm(self._agent_position - p) for p in self._planet_positions],
            dtype=np.float32,
        )

        crash = bool(np.any(self.distance_to_planets <= self.planet_specs[:, 0]))
        out_of_bounds = (
            self._agent_position[0] > self.size
            or self._agent_position[0] < 0
            or self._agent_position[1] > self.size
            or self._agent_position[1] < 0
        )

        terminated = success or crash
        truncated = (self.step_counter >= 200) or out_of_bounds
        reward = 1.0 if success else -1.0 if crash else 0.0

        self.step_counter += 1
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()


# ---------------------------------------------------------------------------
# Registration — continuous env
# ---------------------------------------------------------------------------

try:
    gym.register(
        id="SpaceEnv-v0",
        entry_point=SpaceEnv,
        max_episode_steps=200,
    )
except gym.error.Error:
    pass  # Already registered (e.g. module imported twice)


# ---------------------------------------------------------------------------
# Discrete-action wrapper
# ---------------------------------------------------------------------------

class DiscreteSpaceWrapper(gym.Wrapper):
    """Discretises SpaceEnv's 2-D continuous action space into an n×n grid.

    Each discrete action i maps to a unique (ax, ay) thrust pair drawn from
    an n_bins × n_bins uniform grid over [-max_acceleration, +max_acceleration]².
    The grid is row-major: action i → (bins[i // n_bins], bins[i % n_bins]).

    Also flattens the dict observation to a 1-D float32 array [x, y, vx, vy]
    so that the standard IQN/DQN agents can consume it directly.

    Parameters
    ----------
    env    : a SpaceEnv instance
    n_bins : number of evenly-spaced values per thrust axis (default 5 → 25 actions)
    """

    def __init__(self, env: SpaceEnv, n_bins: int = 5):
        super().__init__(env)
        self.n_bins = n_bins
        a_max = env.max_acceleration
        bins = np.linspace(-a_max, a_max, n_bins, dtype=np.float32)
        # Pre-compute lookup table: discrete index → continuous (ax, ay)
        self._action_map = np.array(
            [(ax, ay) for ax in bins for ay in bins], dtype=np.float32
        )  # shape (n_bins², 2)
        self.action_space = gym.spaces.Discrete(len(self._action_map))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    @staticmethod
    def _flatten(obs: dict) -> np.ndarray:
        return np.concatenate(
            [obs["agent position"], obs["agent velocity"]], dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action: int):
        thrust = self._action_map[int(action)]
        obs, reward, terminated, truncated, info = self.env.step(thrust)
        return self._flatten(obs), reward, terminated, truncated, info


def _make_discrete_space_env(n_bins: int = 5, **kwargs) -> DiscreteSpaceWrapper:
    """Factory registered as 'SpaceEnv-discrete-v0'.

    Extra kwargs (size, planet_specs, sigma_noise, timestep, max_acceleration)
    are forwarded to SpaceEnv so all parameters remain configurable via
    gym.make("SpaceEnv-discrete-v0", n_bins=7, sigma_noise=0.1).
    """
    return DiscreteSpaceWrapper(SpaceEnv(**kwargs), n_bins=n_bins)


try:
    gym.register(
        id="SpaceEnv-discrete-v0",
        entry_point=_make_discrete_space_env,
        max_episode_steps=200,
    )
except gym.error.Error:
    pass  # Already registered


# ---------------------------------------------------------------------------
# Flat-obs wrapper (continuous actions, flattened observations)
# ---------------------------------------------------------------------------

class FlatObsWrapper(gym.Wrapper):
    """Flattens SpaceEnv's dict observation to [x, y, vx, vy], keeping continuous actions.

    Used for the continuous-action mode where IQN/DQN act as critics Q(s,a)→scalar
    and action selection is done by uniform sampling + argmax.
    """

    def __init__(self, env: SpaceEnv):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        # action_space stays as continuous Box from SpaceEnv

    @staticmethod
    def _flatten(obs: dict) -> np.ndarray:
        return np.concatenate(
            [obs["agent position"], obs["agent velocity"]], dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten(obs), reward, terminated, truncated, info


try:
    gym.register(
        id="SpaceEnv-flat-v0",
        entry_point=lambda **kwargs: FlatObsWrapper(SpaceEnv(**kwargs)),
        max_episode_steps=200,
    )
except gym.error.Error:
    pass  # Already registered


# ---------------------------------------------------------------------------
# Data-collection utilities
# ---------------------------------------------------------------------------

def flatten_obs(obs: dict) -> np.ndarray:
    """Convert a dict observation to a flat [x, y, vx, vy] vector."""
    return np.concatenate([obs["agent position"], obs["agent velocity"]]).astype(np.float32)


def get_action(
    policy: str,
    env: SpaceEnv,
    obs: dict,
    planet: np.ndarray = PLANET_CENTER,
) -> np.ndarray:
    """Return a 2-D thrust action for the given policy.

    Policies
    --------
    random   : uniform random action from the action space
    inward   : thrust toward the planet (with small noise)
    outward  : thrust away from the planet (with small noise)
    boundary : thrust tangentially to orbit near r_c (with small noise)
    """
    pos = obs["agent position"]

    if policy == "random":
        return env.action_space.sample()

    direction = planet - pos if policy == "inward" else pos - planet
    norm = np.linalg.norm(direction)
    if norm < 1e-5:
        return env.action_space.sample()

    unit = direction / norm

    if policy in ("inward", "outward"):
        noise = np.random.normal(0, 0.1, size=2)
        action = unit * env.max_acceleration + noise
    elif policy == "boundary":
        # Rotate outward unit 90° to get tangential direction
        outward = (pos - planet) / norm
        unit = np.array([-outward[1], outward[0]])
        noise = np.random.normal(0, 0.15, size=2)
        action = unit * env.max_acceleration + noise
    else:
        raise ValueError(f"Unknown policy: {policy!r}")

    return np.clip(action, -env.max_acceleration, env.max_acceleration).astype(np.float32)


def collect_dataset(
    n_episodes: int = 2000,
    policy_mix: dict = {"random": 0.4, "inward": 0.3, "outward": 0.1, "boundary": 0.2},
    r_c: float = R_C,
    planet: np.ndarray = PLANET_CENTER,
    seed: int = 0,
    planet_specs: list = [[1, 2]],
    sigma_noise: float = 0.05,
    timestep: float = 0.1,
) -> pd.DataFrame:
    """Collect an offline dataset of transitions from SpaceEnv.

    Each transition is labelled ``is_dead_end = True`` when the agent's
    distance to the planet is less than *r_c* at that step.

    Parameters
    ----------
    n_episodes   : number of episodes to collect
    policy_mix   : mapping from policy name to fraction of episodes
                   (must sum to 1.0)
    r_c          : critical radius used for ground-truth dead-end labels
    planet       : planet position array, shape (2,)
    seed         : base random seed for reproducibility
    planet_specs : forwarded to SpaceEnv (list of [radius, GM])
    sigma_noise  : forwarded to SpaceEnv
    timestep     : forwarded to SpaceEnv

    Returns
    -------
    pd.DataFrame with one row per transition and columns:
        episode_id, timestep, x, y, vx, vy, ax, ay, reward,
        next_x, next_y, next_vx, next_vy, done,
        dist_to_planet, is_dead_end, policy
    """
    assert abs(sum(policy_mix.values()) - 1.0) < 1e-6, "policy_mix must sum to 1"

    # Assign a policy to each episode
    episode_policies = []
    for policy, fraction in policy_mix.items():
        episode_policies += [policy] * int(n_episodes * fraction)
    while len(episode_policies) < n_episodes:
        episode_policies.append("random")
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_policies)

    transitions = []

    for episode_id, policy in enumerate(episode_policies):
        env = SpaceEnv(
            planet_specs=planet_specs,
            sigma_noise=sigma_noise,
            timestep=timestep,
        )
        env.reset(seed=seed + episode_id)

        # Randomise start across the full map (excluding planet body)
        planet_radius = env.planet_specs[0][0]
        while True:
            start_x = rng.uniform(0.5, 9.5)
            start_y = rng.uniform(0.5, 9.5)
            if np.linalg.norm(np.array([start_x, start_y]) - planet) > planet_radius + 0.2:
                break
        env._agent_position = np.array([start_x, start_y], dtype=np.float32)
        env._agent_velocity = np.array([0.0, 0.0], dtype=np.float32)

        obs = env._get_obs()
        state = flatten_obs(obs)
        done = False
        t = 0

        while not done:
            action = get_action(policy, env, obs, planet=planet)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            done = terminated or truncated

            dist_to_planet = float(info["distance to planets"][0])
            transitions.append({
                "episode_id":     episode_id,
                "timestep":       t,
                "x":              state[0],
                "y":              state[1],
                "vx":             state[2],
                "vy":             state[3],
                "ax":             action[0],
                "ay":             action[1],
                "reward":         reward,
                "next_x":         next_state[0],
                "next_y":         next_state[1],
                "next_vx":        next_state[2],
                "next_vy":        next_state[3],
                "done":           done,
                "dist_to_planet": dist_to_planet,
                "is_dead_end":    dist_to_planet < r_c,
                "policy":         policy,
            })

            obs = next_obs
            state = next_state
            t += 1

    return pd.DataFrame(transitions)
