# %%
import numpy as np
from gridworld import GridWorld
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

from torchrl.envs.utils import step_mdp
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf
from rl_utils.common import (Evaluator, compress_dict, get_size_for_space,
                             set_seed)
from rl_utils.envs import create_vectorized_envs
from rl_utils.logging import Logger
from tensordict.tensordict import TensorDict


# %%
import sys
sys.path.append("/Users/williamhuang/Documents/Projects/Tobin 2024/Code/code/bc-irl-main") 

from imitation_learning.policy_opt.policy import Policy
from imitation_learning.policy_opt.ppo import PPO
from imitation_learning.policy_opt.storage import RolloutStorage


# %%
world=\
    """
    wwwwww
    wa   w
    w    w
    w    w
    w   gw
    wwwwww
    """
    
env=GridWorld(world,slip=0) # Slip is the degree of stochasticity of the gridworld.

# Value Iteration
V=np.zeros((env.state_count,1))
V_prev=np.random.random((env.state_count,1))
eps=1e-7
gamma=0.9

while np.abs(V-V_prev).sum()>eps:
    Q_sa=env.R_sa+gamma*np.squeeze(np.matmul(env.P_sas,V),axis=2)
    V_prev=V.copy()
    V=np.max(Q_sa,axis=1,keepdims=True)
    pi=np.argmax(Q_sa,axis=1)

print("Pi:",pi)
# env.show(pi)  # Show the policy in graphical window and we can control the agent using the arrow-keys

# %%
value_grid = V.reshape(4, 4)
print("Value Function V(s):\n", value_grid)

# %%
# Create the heatmap
plt.imshow(value_grid, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Value")

# Annotate each cell with its value
for i in range(value_grid.shape[0]):
    for j in range(value_grid.shape[1]):
        plt.text(j, i, f"{value_grid[i, j]:.2f}", ha="center", va="center", color="white")

# Label and show the plot
plt.title("True Value Grid")
plt.xlabel("X-axis (columns)")
plt.ylabel("Y-axis (rows)")
plt.show()

# %%
def set_seed(seed: int) -> None:
    """
    Sets the seed for numpy, python random, and pytorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# %%
from gymnasium import Env
class vectorized_env():
    def __init__(self, envs : List[Env]):
        self.envs = envs
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):

        return torch.tensor([env.reset()[0].tolist() for env in self.envs],dtype=torch.float32)
    
    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        steps = [env.step(action[i]) for i,env in enumerate(self.envs)]
        return_value = (torch.tensor([step[0].tolist() for step in steps],dtype=torch.float32),
                torch.tensor([step[1] for step in steps],dtype=torch.float32),
                torch.tensor([step[2] for step in steps],dtype=torch.bool),
                [step[3] for step in steps])
        return return_value

# %%
cfg = yaml.load(open("bc-irl-mouse.yaml", 'r'), Loader=yaml.SafeLoader)
cfg = DictConfig(cfg)

# %%
set_seed(cfg.seed)
device = torch.device(cfg.device)

# Setup the environments
envs = vectorized_env([GridWorld(world, slip=0, isDRL=True) for _ in range(cfg.num_envs)])

steps_per_update = cfg.num_steps * cfg.num_envs

num_updates = int(cfg.num_env_steps) // steps_per_update

# Set dynamic variables in the config.
cfg.obs_shape = envs.observation_space.shape
cfg.action_dim = envs.action_space.n
cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)
cfg.total_num_updates = num_updates

logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)
print("policy",cfg.policy)
policy = hydra_instantiate(cfg.policy)
policy = policy.to(device)
print("policy_updater",cfg.policy_updater)
updater = hydra_instantiate(cfg.policy_updater, policy=policy, device=device).to(device)


start_update = 0
if cfg.load_checkpoint is not None:
    # Load a checkpoint for the policy/reward. Also potentially resume
    # training.
    ckpt = torch.load(cfg.load_checkpoint)
    updater.load_state_dict(ckpt["updater"], should_load_opt=cfg.resume_training)
    if cfg.load_policy:
        policy.load_state_dict(ckpt["policy"])
    if cfg.resume_training:
        start_update = ckpt["update_i"] + 1

eval_info = {"run_name": logger.run_name}

import warnings 
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Storage for the rollouts
obs = envs.reset()
td = TensorDict({"observation": obs}, batch_size=[cfg.num_envs])

# Storage for the rollouts
storage_td = TensorDict({}, batch_size=[cfg.num_envs, cfg.num_steps], device=device)

for update_i in range(start_update, num_updates):
    is_last_update = update_i == num_updates - 1
    for step_idx in range(cfg.num_steps):

        # Collect experience.
        with torch.no_grad():
            policy.act(td)
        next_obs, reward, done, infos = envs.step(td["action"])

        td["next_observation"] = next_obs
        for env_i, info in enumerate(infos):
            if "final_obs" in info:
                td["next_observation"][env_i] = info["final_obs"]
        td["reward"] = reward.reshape(-1, 1)
        td["done"] = done
    
        storage_td[:, step_idx] = td
        # Log to CLI/wandb.
        logger.collect_env_step_info(infos)
    
    # Call method specific update function
    updater.update(policy, storage_td, logger, envs=envs)

    if cfg.log_interval != -1 and (
        update_i % cfg.log_interval == 0 or is_last_update
    ):
        logger.interval_log(update_i, steps_per_update * (update_i + 1))
        
        # Use GridWorld environment for evaluation
        eval_env = GridWorld(cfg.env.env_settings.params.config)
        
        height = 2
        width = 2
        fig, ax = plt.subplots(nrows=height, ncols=width, sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0})

        last_reward_map = np.zeros((eval_env.col, eval_env.row))

        for i in range(height):
            for j in range(width):
                
                reward_map = np.zeros((eval_env.col, eval_env.row))
                agent_pos = eval_env.reset()
                # Get the position of the agent and the goal (or other states)
                for x in range(eval_env.col):
                    for y in range(eval_env.row):
                        state = eval_env.state_dict.get((x, y))
                        # Reward calculation logic (example: for goal and hole)
                        reward_map[x, y] = updater.reward(
                            next_obs = torch.tensor([*agent_pos, x, y] + [0]*(eval_env.observation_space.shape[0] - 4), dtype=torch.float32).to(device).view(1, 1, -1)
                        )
                
                # Define the color map
                cmap = plt.cm.get_cmap('hot')

                # Plot the reward map without axis and numbers
                image = ax[i,j].imshow(reward_map, cmap=cmap, interpolation='nearest')
                ax[i,j].axis('off')

                # Plot the agent's position (use agent's current position)
                ax[i,j].scatter(
                    agent_pos[1] * eval_env.row, 
                    agent_pos[0] * eval_env.col, 
                    c='blue', 
                    s=60
                )
                
                last_reward_map = reward_map
                
        plt.tight_layout()
        plt.savefig(osp.join(logger.save_path, f"reward_map.{update_i}.png"))
        print(f"Saved to {osp.join(logger.save_path, f'reward_map.{update_i}.png')}")

    if cfg.save_interval != -1 and (
        (update_i + 1) % cfg.save_interval == 0 or is_last_update
    ):
        save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
        torch.save(
            {
                "policy": policy.state_dict(),
                "updater": updater.state_dict(),
                "update_i": update_i,
            },
            save_name,
        )
        print(f"Saved to {save_name}")
        eval_info["last_ckpt"] = save_name

logger.close()
print(eval_info)

# updater = hydra_instantiate(cfg.policy_updater, policy=policy, device=device).to(device)
#     set_env_settings = {
#         k: hydra_instantiate(v) if isinstance(v, DictConfig) else v
#         for k, v in cfg.env.env_settings.items()
#     }
#     envs = create_vectorized_envs(
#         cfg.env.env_name,
#         cfg.num_envs,
#         seed=cfg.seed,
#         device=device,
#         **set_env_settings,
#     )

#     steps_per_update = cfg.num_steps * cfg.num_envs
#     num_updates = int(cfg.num_env_steps) // steps_per_update

#     cfg.obs_shape = envs.observation_space.shape
#     cfg.action_dim = get_size_for_space(envs.action_space)
#     cfg.action_is_discrete = isinstance(cfg.action_dim, spaces.Discrete)
#     cfg.total_num_updates = num_updates

#     logger: Logger = hydra_instantiate(cfg.logger, full_cfg=cfg)

#     storage: RolloutStorage = hydra_instantiate(cfg.storage, device=device)
#     policy: Policy = hydra_instantiate(cfg.policy)
#     policy = policy.to(device)
#     updater = hydra_instantiate(cfg.policy_updater, policy=policy, device=device)
#     evaluator: Evaluator = hydra_instantiate(
#         cfg.evaluator,
#         envs=envs,
#         vid_dir=logger.vid_path,
#         updater=updater,
#         logger=logger,
#         device=device,
#     )

#     start_update = 0
#     if cfg.load_checkpoint is not None:
#         ckpt = torch.load(cfg.load_checkpoint)
#         updater.load_state_dict(ckpt["updater"], should_load_opt=cfg.resume_training)
#         if cfg.load_policy:
#             policy.load_state_dict(ckpt["policy"])
#         if cfg.resume_training:
#             start_update = ckpt["update_i"] + 1

#     eval_info = {"run_name": logger.run_name}

#     if cfg.only_eval:
#         eval_result = evaluator.evaluate(policy, cfg.num_eval_episodes, 0)
#         logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
#         eval_info.update(eval_result)
#         logger.interval_log(0, 0)
#         logger.close()

#         return eval_info

#     obs = envs.reset()
#     storage.init_storage(obs)

#     for update_i in range(start_update, num_updates):
#         is_last_update = update_i == num_updates - 1
#         for step_idx in range(cfg.num_steps):
#             with torch.no_grad():
#                 act_data = policy.act(
#                     storage.get_obs(step_idx),
#                     storage.recurrent_hidden_states[step_idx],
#                     storage.masks[step_idx],
#                 )
#             next_obs, reward, done, info = envs.step(act_data["actions"])
#             storage.insert(next_obs, reward, done, info, **act_data)
#             logger.collect_env_step_info(info)

#         updater.update(policy, storage, logger, envs=envs)

#         storage.after_update()

#         if cfg.eval_interval != -1 and (
#             update_i % cfg.eval_interval == 0 or is_last_update
#         ):
#             with torch.no_grad():
#                 eval_result = evaluator.evaluate(
#                     policy, cfg.num_eval_episodes, update_i
#                 )
#             logger.collect_infos(eval_result, "eval.", no_rolling_window=True)
#             eval_info.update(eval_result)

#         if cfg.log_interval != -1 and (
#             update_i % cfg.log_interval == 0 or is_last_update
#         ):
#             logger.interval_log(update_i, steps_per_update * (update_i + 1))

#         if cfg.save_interval != -1 and (
#             (update_i + 1) % cfg.save_interval == 0 or is_last_update
#         ):
#             save_name = osp.join(logger.save_path, f"ckpt.{update_i}.pth")
#             torch.save(
#                 {
#                     "policy": policy.state_dict(),
#                     "updater": updater.state_dict(),
#                     "update_i": update_i,
#                 },
#                 save_name,
#             )
#             print(f"Saved to {save_name}")
#             eval_info["last_ckpt"] = save_name

#     logger.close()
#     return eval_info


# if __name__ == "__main__":
#     main()


