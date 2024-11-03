# Importing Packages
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from imitation.algorithms.adversarial.airl import AIRL
from imitation.util import util
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.rewards.reward_nets import RewardNet
from imitation.util.networks import RunningNorm
from imitation.util import networks, util
import matplotlib.pyplot as plt
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.init as init
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import util

# Define and initialize the environment
class GameEnv(gym.Env):
    def __init__(self):
        self.height = 4
        self.width = 4
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(self.height, self.width)-1, shape=(2,), dtype=np.int32)

        self.moves = {
            0: (-1, 0),   # up
            1: (0, 1),    # right
            2: (1, 0),    # down
            3: (0, -1)    # left
        }

        self.true_rewards = np.array([
            [0, -1, -1, -1],
            [-1, 0, -1, 0],
            [-1, -1, -1, -1],
            [-1, -1, -1, 10]
        ])
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.S = (0, 0)  # Start position
        return np.array(self.S, dtype=np.int32)  # Only return the initial observation

    def step(self, action):
        # Calculate new position based on the action
        dx, dy = self.moves[action]
        self.S = (self.S[0] + dx, self.S[1] + dy)

        # Enforce grid boundaries
        self.S = (max(0, min(self.S[0], self.height - 1)),
                max(0, min(self.S[1], self.width - 1)))

        # Calculate reward based on new position
        reward = self.true_rewards[self.S]

        # Check if goal (bottom-right corner) is reached
        done = self.S == (self.height - 1, self.width - 1)
        truncated = False  # No truncation conditions in this environment
        info = {"obs": np.array(self.S), "rews": reward}

        # Return new observation, reward, done, truncated, and info
        return np.array(self.S, dtype=np.int32), reward, done, truncated, info

# Step 1: Wrap the environment
env = DummyVecEnv([lambda: GameEnv()])

# Step 2: Collect expert demonstrations using a trained PPO agent
SEED = 42
expert = PPO("MlpPolicy", env, verbose=1, seed=SEED)
expert.learn(total_timesteps=2000)
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_episodes=500),
    rng=np.random.default_rng(SEED),
)

# Step 3: Define and initialize the learner, reward network, and AIRL trainer
learner = PPO(
    "MlpPolicy",
    env=env,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=SEED,
)

reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm
)

airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

# Train AIRL
airl_trainer.train(total_timesteps=2000)

