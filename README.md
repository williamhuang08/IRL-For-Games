# Inverse Reinforcement Learning for eBooks

## Models

### AIRL for eBooks with simulated data
We begin by implementing Adversarial Inverse Reinforcement Learning (AIRL) to learn an underlying reward function in a hypothetical e-reading service environment. In essence, we build a true reward function and Markov decision process (MDP) environment. Then, an agent policy is trained in the environment and trajectories are collected. Using these (state, action) pairs over the agent's trajectories, we evaluate whether the reward function is able to match "expert" behavior.

#### 1. Model Design
Let score = a * engagement_level + b * section_number, where a > 0, b > 0 and engagement_level and section_number are normalized to a number between 0 and 1. Arbitrarily, we let a = 1 and b = 0.5.

As in typical inverse reinforcement learning environments, we define a true reward function r(s, a), which evaluates the reward a user gets if he takes action a for a given state **[section_number, engagement_level, time (# number of hours since last engagement), price]** to an action **[wait, read_without_payment, read_with_payment]**

```
    if score > theta_1, continue reading (including paying):
      if price = 1, read with paying (r = 5), wait (r = -1)
      if price = 0, read without paying (r = 9), wait (r = -1)
    else if theta_2 < score <= theta_1, 
      if time since last read < 24 hours:
          if price = 1, read with paying (r = 4), wait (r = -1)
          if price = 0, read (r = 7), wait (r = -1)
      else: # continue reading only if free (do not pay)
          if price = 1, read with paying (r = 3), wait (r = -1)
          if price = 0, read (r = 6), wait (r = -1)
    else:
      if  time since last read < 36 hours, continue reading only if free (do not pay) 
          if price = 1, read with paying (r = 2), wait (r = -1)
          if price = 0, read (r = 5), wait (r = -1)
      else: wait (no reading)
          if price = 1, read with paying (r = 1), wait (r = -1)
          if price = 0, read (r = 3), wait (r = -1)
```
Transition function from states to states
```
P(s_{t+1}|s_{t},a) = p(section_number_{t+1}, engagement_level_{t+1}, time_{t+1}, price_{t+1}|s_t, a) = p(section number_{t+1}, engagement_level_{t+1}, time_{t+1}|s_t, a) * p(price_{t+1}|s_t,a)
```

Here, price is a function of sections, price = 1 with probability that increases with section number  
P(price_{t+1}) =  1 - \gamma^{section number_t} (\gamma = 0.9) (drawn from bernouli distribution)  
time_{t+1} = time_{t} + 1 hour  
engagement_level_{t+1} = drawn from a normal distribution  
section_number_{t+1} = section_number_{t} +1 only if the previous action is "read" (buy or not buy), otherwise, it equals section_number_{t}  

When the action is "read" (buy or free), reset time to 1 for the next state, because next decision-making point is 1 hour after reading the section.

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