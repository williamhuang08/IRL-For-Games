
import gymnasium as gym
import numpy as np
import pygame as pg
from collections import deque
from gym.spaces import Box,Discrete
from collections import defaultdict
from gridworld.modules import Agent, Wall, Goal, State, Hole, Block
import numpy as np
import matplotlib.pyplot as plt
import torch
import os.path as osp
import random
from typing import Dict,List
import gym.spaces as spaces
import hydra
import numpy as np
from typing import Tuple
import yaml
from collections import defaultdict
from typing import Dict, Optional

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf
from rl_utils.common import (Evaluator, compress_dict, get_size_for_space,
                             set_seed)
from rl_utils.envs import create_vectorized_envs
from rl_utils.logging import Logger


import numpy as np
from gym.utils import seeding

class GridWorld(gym.Env):
    def __init__(self, grid_size=(5, 5), start_position=(0, 0), goal_position=(4, 4), obstacles=None):
        super(GridWorld, self).__init__()
        
        # Initialize environment parameters
        self.grid_size = grid_size
        self.start_position = start_position
        self.goal_position = goal_position
        self.obstacles = obstacles if obstacles else []
        
        # Define action and observation spaces
        # Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.action_space = spaces.Discrete(4)
        # Observation: Agent's current position in the grid (x, y)
        self.observation_space = spaces.Box(low=0, high=max(grid_size)-1, shape=(2,), dtype=np.int32)
        
        # Initialize agent's position
        self.state = np.array(self.start_position)
        
        # Initialize the random seed
        self.seed()

    def seed(self, seed=None):
        # Seed the environment's random number generator
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Reset the agent's position to the start
        self.state = np.array(self.start_position)
        return self.state

    def step(self, action):
        # Define movement directions
        movement = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        # Calculate the new position
        new_position = self.state + np.array(movement[action])
        
        # Check if the new position is within the grid bounds
        if (0 <= new_position[0] < self.grid_size[0]) and (0 <= new_position[1] < self.grid_size[1]):
            # Check if the new position is not an obstacle
            if tuple(new_position) not in self.obstacles:
                self.state = new_position
        
        # Check if the agent has reached the goal
        done = np.array_equal(self.state, self.goal_position)
        reward = 1 if done else -0.1  # Reward for reaching goal, penalty otherwise

        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Render the grid
        grid = np.full(self.grid_size, ' ')
        grid[self.goal_position] = 'G'  # Goal
        for obs in self.obstacles:
            grid[obs] = 'X'  # Obstacles
        grid[tuple(self.state)] = 'A'  # Agent
        print("\n".join(["".join(row) for row in grid]))
        print()

    def close(self):
        pass



def set_seed(seed: int) -> None:
    """
    Sets the seed for numpy, python random, and pytorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


cfg = yaml.load(open("bc-irl-mouse.yaml", 'r'), Loader=yaml.SafeLoader)
cfg = DictConfig(cfg)

from gym.envs.registration import register

register(
    id='mouse-v0',
    entry_point='__main__:GridWorld',  
)

set_seed(cfg.seed)
device = torch.device(cfg.device)

set_env_settings = {
    k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
    for k, v in cfg.env.env_settings.items()
}

envs = create_vectorized_envs(
    cfg.env.env_name,
    cfg.num_envs,
    seed=cfg.seed,
    device=device,
    **set_env_settings,
)