import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from collections import deque

@dataclass
class DeadEnd:
    center: np.ndarray
    type: str  #'disc' or 'ellipse'
    death_a: float  #Death zone semi-axis a
    death_b: float  #Death zone semi-axis b (same as semi-axis a for a disc)
    dead_end_a: float   #Dead End zone semi-axis a
    dead_end_b: float   #Dead End zone semi-axis b (same as semi-axis a for a disc)
    theta: float = 0.0  #Ellipse rotation angle, not useful for discs

class MedGridGeneralEnv(gym.Env):
    """10x10 continuous gridworld with procedural dead-end generation.

    State:  (x, y) ∈ [0, 10]²
    Action: (x_target, y_target) ∈ [0, 10]² — desired next position.
    Start:  (10, 0) — bottom-right corner.

    Dead end mechanics:
        If the current state is in a dead-end,
        the next state is forced to be inside the corresponding death zone
    """
    
    SIZE = 10.0
    START = np.array([10.0, 0.0], dtype=np.float32)
    MAX_STEPS = 200
    
    _RECOVERY_RADIUS = 0.5
    _RECOVERY_CENTER = np.array([7.5, 8.0], dtype=np.float32)

    def __init__(self, num_dead_ends: int = 3, scale: float = 1.0, seed: int = None):
        super().__init__()
        
        self.scale = scale
        self.num_dead_ends = num_dead_ends
        
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
        self.danger_zones = []
        
        # Generate the map layout during initialization 
        # (This keeps the layout static for offline dataset collection)
        self._generate_map(seed)

    def _generate_map(self, seed: int = None, max_attempts: int = 2000):
        """Randomly generates n valid dead-ends by checking their validity"""
        if seed is not None:
            np.random.seed(seed)
            
        self.danger_zones = []
        attempts = 0
        
        #While we don't have n dead-ends and we are not exceeding the maximum number of generation attempts
        while len(self.danger_zones) < self.num_dead_ends and attempts < max_attempts:
            attempts += 1
            
            #Random center's 2D coordinates
            center = np.random.uniform(0, self.SIZE, size=2)
            shape_type = np.random.choice(['circle', 'ellipse'])
            
            #Random size, shape, and orientation of the death and dead-end zones
            if shape_type == 'circle':
                death_r = np.exp(np.random.uniform(np.log10(0.25), np.log(2.5))) * self.scale
                dead_end_r = death_r + np.random.uniform(0.5, 1.5) * self.scale
                zone = DeadEnd(center, 'circle', death_r, death_r, dead_end_r, dead_end_r)
            else:
                death_a = np.exp(np.random.uniform(np.log10(0.25), np.log(2.5))) * self.scale
                death_b = np.exp(np.random.uniform(np.log10(0.25), np.log(2.5))) * self.scale
                dead_end_a = death_a + np.random.uniform(0.5, 1.5) * self.scale
                dead_end_b = death_b + np.random.uniform(0.5, 1.5) * self.scale
                theta = np.random.uniform(0, 2 * np.pi)
                zone = DeadEnd(center, 'ellipse', death_a, death_b, dead_end_a, dead_end_b, theta)

            #Checking whether the generated
            if self._is_valid_placement(zone):
                self.danger_zones.append(zone)
                
        if len(self.danger_zones) < self.num_dead_ends:
            raise RuntimeError(f"Could only generate {len(self.danger_zones)} out of {self.num_dead_ends} valid zones. Try a smaller number or lower scale.")

    def _is_valid_placement(self, new_zone):
        """Ensures start and recovery are not in dead ends or death zones,
        and a path exists between them."""
        current_zones = self.danger_zones + [new_zone]
        
        #Start and recovery clearance
        if self._check_collision(self.START, current_zones)[0] != "neutral": return False
        if self._check_collision(self._RECOVERY_CENTER, current_zones)[0] not in ["neutral", "recovery"]: return False

        #Pathfinding Check (Discretize grid to 20x20 for rough BFS)
        grid_res = 20
        grid = np.zeros((grid_res, grid_res), dtype=bool)
        
        #Checking the state at every point of the grid
        for i in range(grid_res):
            for j in range(grid_res):
                x = (i / (grid_res - 1)) * self.SIZE
                y = (j / (grid_res - 1)) * self.SIZE
                status, _ = self._check_collision(np.array([x, y]), current_zones)
                if status in ["death", "dead_end"]:
                    grid[i, j] = True

        #Map coordinates to grid indices
        start_idx = (int(self.START[0] / self.SIZE * (grid_res-1)), 
                     int(self.START[1] / self.SIZE * (grid_res-1)))
        goal_idx = (int(self._RECOVERY_CENTER[0] / self.SIZE * (grid_res-1)), 
                    int(self._RECOVERY_CENTER[1] / self.SIZE * (grid_res-1)))

        #The starting point or the recovery point should not be in a dead-end or a death zone
        if grid[start_idx] or grid[goal_idx]: return False

        #Breadth-First-Search
        queue = deque([start_idx])
        visited = set([start_idx])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        while queue:
            cx, cy = queue.popleft()

            #If a path between the starting point and recovery exists,
            #return true, otherwise the generated map is not relevant
            if (cx, cy) == goal_idx:
                return True
                
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_res and 0 <= ny < grid_res:
                    if not grid[nx, ny] and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        
        return False

    def _check_collision(self, state, zones):
        """Returns (status, dead_end_center_if_applicable). Status: 'death', 'dead-end', 'recovery', 'neutral'"""
        for zone in zones:
            dx = state[0] - zone.center[0]
            dy = state[1] - zone.center[1]
            
            cos_t = np.cos(-zone.theta)
            sin_t = np.sin(-zone.theta)

            #"Unrotating" the point to check if it is inside the rotated ellipse 
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            
            death_val = (rx**2 / zone.death_a**2) + (ry**2 / zone.death_b**2)
            dead_end_val = (rx**2 / zone.dead_end_a**2) + (ry**2 / zone.dead_end_b**2)
            
            if death_val <= 1.0:
                return "death", None
            elif dead_end_val <= 1.0:
                return "dead_end", zone.center.copy()
                
        dist_recovery = float(np.sqrt((state[0] - self._RECOVERY_CENTER[0]) ** 2 + (state[1] - self._RECOVERY_CENTER[1]) ** 2))
        if dist_recovery <= self._RECOVERY_RADIUS * self.scale:
            return "recovery", None
            
        return "neutral", None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #We don't regenerate the map on reset to keep data collection consistent.
        #Create a new environment instance for a new map.
        
        if options is not None and "start_state" in options:
            self.state = np.array(options["start_state"], dtype=np.float32)
        else:
            self.state = self.START.copy()
            
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        #Determine next intended state
        status, dead_end_center = self._check_collision(self.state, self.danger_zones)
        
        if status == "dead_end":
            #Override action: teleport to the center of the dead end
            next_state = dead_end_center
        else:
            next_state = np.clip(np.array(action, dtype=np.float32), 0.0, self.SIZE)
            
        self.state = next_state
        self.steps += 1

        #Evaluate new state
        new_status, _ = self._check_collision(self.state, self.danger_zones)
        
        if new_status == "death":
            return self.state.copy(), -1.0, True, False, {}
        if new_status == "recovery":
            return self.state.copy(), +1.0, True, False, {}
            
        truncated = self.steps >= self.MAX_STEPS
        return self.state.copy(), 0.0, False, truncated, {}


try:
    gym.register(id="MedGridGeneral-v0", entry_point="med_grid_general_env:MedGridGeneralEnv")
except Exception:
    pass


class DiscreteMedGridGeneralWrapper(gym.Wrapper):
    """Discretises MedGridGeneral's 2-D continuous action space into an n×n grid."""

    def __init__(self, env: MedGridGeneralEnv, n_bins: int = 5):
        super().__init__(env)
        bins = np.linspace(0.0, MedGridGeneralEnv.SIZE, n_bins, dtype=np.float32)
        self._action_map = np.array(
            [(x, y) for x in bins for y in bins], dtype=np.float32
        )  
        self.action_space = gym.spaces.Discrete(len(self._action_map))

    def step(self, action: int):
        target = self._action_map[int(action)]
        return self.env.step(target)

def _make_discrete_medgridgeneral(n_bins: int = 5, num_dead_ends: int = 3, scale: float = 1.0, seed: int = None) -> DiscreteMedGridGeneralWrapper:
    env = MedGridGeneralEnv(num_dead_ends=num_dead_ends, scale=scale, seed=seed)
    return DiscreteMedGridGeneralWrapper(env, n_bins=n_bins)

try:
    gym.register(id="MedGridGeneral-discrete-v0", entry_point="med_grid_general_env:_make_discrete_medgridgeneral")
except Exception:
    pass