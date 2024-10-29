# Inverse Reinforcement Learning for Games

## Folder Organization

### bc-irl-main

The bc-irl-main folder contains the project code from Meta Research's [BC-IRL project](https://github.com/facebookresearch/bc-irl). Please see their Github and paper for more detailed information regarding their model and implementation. 

Core Folders
- **bc-irl-main/data/**
    - checkpoints: If run.py is ran, the reward and policy will be saved in a folder to this location.
    - vids: If run.py is ran, the plotted learned reward function in the environment will be here.
- **bc-irl-main/imitation_learning/bc_irl/** (there are many other models (airl, gail, gl, f_irl, but the focus of our study is bc-irl))
    - differentiable_ppo.py: DifferentiablePPO is a differentiable PPO algorithm aimed to update the policy based on interactions with the environment. The compute_derived method gets the advantages and returns, which are necessary for policy updates in update method.
    - rewards.py: This file contains the class StructuredReward, which incentivizes an agent to go toward the center. GTReward gives smaller rewards as the agent moves further from the goal.
    - updated.py: This file contains the BCIRL class to perform the meta learning where PPO is used to update the policy and then the IRL loss is used to update the reward
- **bc-irl-main/imitation_learning/common/**
    - net.py: Contains the learnable reward function (a neural network)
    - plotting.py: Plots the expert vs. learned reward action
    - pointmass_utils.py: Contains several reward functions and visualizations for the Point Mass environment
- **bc-irl-main/imitation_learning/config/**
    - default.yaml: This configuration is used if no config is provided
- **bc-irl-main/imitation_learning/config/bc_irl/**
    - pointmass_obstacle: Contains the bc-irl model parameters for the 2D pointmass example with obstacles
    - pointmass: Contains the bc-irl model parameters for the 2D pointmass example wihtout obstacles
- **bc-irl-main/imitation_learning/config/env/**
    - Contains the YAML files to specify environment parameters
- **bc-irl-main/imitation_learning/policy_opt/**
    - policy.py: This file handles creates a Policy class to handle both discrete (Categorical class) and continuous action spaces (DiagGaussian). It learns the policy and value function for advantage estimation
    - ppo.py: Another implementation of PPO, but is not differentiable and not suitable for differentiable frameworks like meta-learning
    - storage.py: Saves the rollouts from the expert policy for on-policy algorithms like PPO
- **bc-irl-main/imitation_learning/**
    - run.py: Learns a reward function according to the config (configures policy, optimizer, etc)




## Citation
@article{szot2023bc,
  title={BC-IRL: Learning Generalizable Reward Functions from Demonstrations},
  author={Szot, Andrew and Zhang, Amy and Batra, Dhruv and Kira, Zsolt and Meier, Franziska},
  journal={arXiv preprint arXiv:2303.16194},
  year={2023}
}