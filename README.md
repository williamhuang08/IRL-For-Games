# Inverse Reinforcement Learning to Decode Human Behavior within eReaders

## Models

### AIRL for eBooks with simulated data
We begin by implementing Adversarial Inverse Reinforcement Learning (AIRL) to learn an underlying reward function in a hypothetical e-reading service environment. In essence, we build a true reward function and Markov decision process (MDP) environment. Then, the agent policy is run in the environment and trajectories are collected. Using these (state, action) pairs, we train an AIRL model where the generator is a PPO policy and the discriminator identifies expert trajectories from trajectories generated by the PPO policy. Then, using the original agent's trajectories, we evaluate whether the learned policy and reward function is able to match "expert" behavior.

#### 1. Model Design
Let score = a * engagement_level + b * section_number, where a > 0, b > 0 and engagement_level and section_number are normalized to a number between 0 and 1. Arbitrarily, we let a = 1 and b = 0.5.

As in typical inverse reinforcement learning environments, we define a true reward function r(s, a), which evaluates the reward a user gets if he takes action a for a given state **[section_number, engagement_level, time (# number of hours since last engagement), price (whether or not the agent has to pay to read at this moment), wff (whether or not the current chapter is wait-for-free), wff_hours_required (hours reader must wait to read), wff_hours_waited (hours waited by the reader so far)]** to an action **[wait, read_without_payment, read_with_payment]**

##### Reward r: (s, a) -> R

```
score = interest_score(engagement_level, section_number)
if section is beginning section:
    if score > theta_1:
        if price == 0: # price will always be 0 since the beginning is free
            if action == ACTION_READ_FREE:
                return 9
            else: # Wait
                return -1
	elif theta_2 < score < theta_1:
		if price == 0: # price will always be 0 since the beginning is free
			if action == ACTION_READ_FREE:
				return 7
			else: # Wait
				return -1
	else: # time interval since last read is long (reader is less engaged)
		if price == 0: # price will always be 0 since the beginning is free
			if action == ACTION_READ_FREE:
				return 5
			else: # Wait
				return -1
if section is middle section:
    if current section is wff:
        return 6 * (1 - gamma) ** wff_hours_waited (in days)
    else: # reader either buys or waits
        if score > theta_1:
            if price == 1: # price will always be 1 since the chapter has a price
                if action == ACTION_READ_BUY:
                    return 10
                else: # Wait
                    return -1
        elif theta_2 < score < theta_1:
            if price == 0: # price will always be 1 since the chapter has a price
                if action == ACTION_READ_BUY:
                    return 8
                else: # Wait
                    return -1
        else: # time interval since last read is long (reader is less engaged)
            if price == 0: # price will always be 1 since the chapter has a price
                if action == ACTION_READ_BUY:
                    return 6
                else: # Wait
                    return -1
if section is end section:
    if score > theta_1:
        if price == 1: # price will always be 0 since the end has a price
            if action == ACTION_READ_BUY:
                return 11
            else: # Wait
                return -1
	elif theta_2 < score < theta_1:
		if price == 0: # price will always be 0 since the end has a price
			if action == ACTION_READ_BUY:
				return 9
			else: # Wait
				return -1
	else: # time interval since last read is long (reader is less engaged)
		if price == 0: # price will always be 0 since the end has a price
			if action == ACTION_READ_BUY:
				return 7
			else: # Wait
				return -1
```

##### Transition function from states to states
```
P(s_{t+1}|s_{t},a) = p(section_number_{t+1}, engagement_level_{t+1}, time_{t+1}, wff{t+1}, wff_hours_required{t+1}, wff_hours_waited{t+1}, price_{t+1}|s_t, a),  = p(section number_{t+1}, engagement_level_{t+1}, time_{t+1}, wff{t+1}, wff_hours_required{t+1}, wff_hours_waited{t+1}, |s_t, a) * p(price_{t+1}|s_t,a)
```

```
if beginning chapters are entered:
    if action == ACTION_READ_FREE: 
        section_number += 1
        time = 0 
    price = 0
    engagement_level drawn from normal
    wff = 0
    wff_hours = 0
if middle chapters are entered:
    if current chapter is not wff (reader must buy):
        if ACTION_READ_BUY:
            section_number += 1
            if section_number is middle chapter:
                wff drawn from bernoulli
                if wff:
                    wff_hours_required set to number between 1-72 (uniform distribution)
                    wff_hours_waited = 0
            engagement_level drawn from normal
            time = 0
        time += 1
        price = 1
    else (current chapter is wff):
        if ACTION_READ_BUY: # case where the agent decides the buy instead of keep waiting
            wff = 0
            wff_hours_required = 0
            wff_hours_waited = 0
            time += 1
        else:
            wff_hours -= 1
            if wff_hours == 0:
                wff = 0
if end chapters are entered:
    if action == ACTION_READ_BUY: 
        section_number += 1
        time = 0 
    price = 1
    engagement_level drawn from normal
    wff = 0
    wff_hours = 0       
done = time >= 108 or section_number > NUM_CHAPTERS
new_state = [section_number, engagement_level, time, price, wff, wff_hours]
return new_state
    
```
Note:  
- When the action is "read" (buy or free), reset time to 1 for the next state, because next decision-making point is 1 hour after reading the section.  
- For the first few chapters, the price will always be 0 
- For the middle chapters, the wait-for-free concept is implemented
- For the last few chapters, the price will always be 1 
- Terminating Condition: when the time interval > 108 (we assume that after a large amount of time, you probably won't read anymore) or you have reached the end of the book
<!-- ##### Policy π: S -> A
```
If score > theta_1: continue reading (including paying) with high probabilities
    if price = 1:
        action = ACTION_READ_W_PAY (p = 0.7), WAIT (p = 0.3)
    if price = 0:
        action = ACTION_READ_WO_PAY (p = 0.9), WAIT (p = 0.1)
Else if theta_2 < score < theta_1: 
    if time < 24: # time interval since last read is short
        if price = 1:
            action = ACTION_READ_W_PAY (p = 0.5), WAIT (p = 0.5)
        if price = 0:
            action = ACTION_READ_WO_PAY (p = 0.7), WAIT (p = 0.3)	
    else: # time interval since last read is long (reader is less engaged)
        if price = 1:
            action = ACTION_READ_W_PAY (p = 0.3), WAIT (p = 0.7)
        if price = 0:
            action = ACTION_READ_WO_PAY (p = 0.5), WAIT (p = 0.5)				
Else: in all cases, the probability of reading with and without paying is low
    if time < 36: # time interval since last read is short
        if price = 1:
            action = ACTION_READ_W_PAY (p = 0.2), WAIT (p = 0.8)
        if price = 0:
            action = ACTION_READ_WO_PAY (p = 0.4), WAIT (p = 0.6)	
    else: # time interval since last read is long (reader is less engaged)
        if price = 1:
            action = ACTION_READ_W_PAY (p = 0.1), WAIT (p = 0.9)
        if price = 0:
            action = ACTION_READ_WO_PAY (p = 0.2), WAIT (p = 0.8)		
return action		
```
##### Reward r: (s, a) -> R
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
``` -->

<!-- 
Here, price is a function of sections, price = 1 with probability that increases with section number  
P(price_{t+1}) =  1 - \gamma^{section number_t} (\gamma = 0.7) (drawn from bernouli distribution)  
time_{t+1} = time_{t} + 1 hour  
engagement_level_{t+1} = drawn from a normal distribution  
section_number_{t+1} = section_number_{t} +1 only if the previous action is "read" (buy or not buy), otherwise, it equals section_number_{t}   -->


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