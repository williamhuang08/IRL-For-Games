# Inverse Reinforcement Learning for Games

## Folder Organization

**bc-irl-main**

The bc-irl-main folder contains the project code from Meta Research's [BC-IRL project](https://github.com/facebookresearch/bc-irl). Please see their Github and paper for more detailed information regarding their model and implementation. 

bc-irl-main/.hydra: You can ignore this folder.

- bc-irl-main/data:
    - checkpoints: If run.py is ran, the reward and policy will be saved in a folder to this location.
    - vids: If run.py is ran, the plotted learned reward function in the environment will be here.
- imitation_learning/bc_irl (there are many other models (airl, gail, gl, f_irl, but the focus of our study is bc-irl))
    - differentiable_ppo.py: DifferentiablePPO is a differentiable PPO algorithm aimed to update the policy based on interactions with the environment. The compute_derived method gets the advantages and returns, which are necessary for policy updates in update method.
    - rewards.py: This file contains the class StructuredReward, which incentivizes an agent to go toward the center. GTReward gives smaller rewards as the agent moves further from the goal.
    - updated.py: This file contains the BCIRL class to perform the meta learning where PPO is used to update the policy and then the IRL loss is used to update the reward
- 


## Citation
@article{szot2023bc,
  title={BC-IRL: Learning Generalizable Reward Functions from Demonstrations},
  author={Szot, Andrew and Zhang, Amy and Batra, Dhruv and Kira, Zsolt and Meier, Franziska},
  journal={arXiv preprint arXiv:2303.16194},
  year={2023}
}