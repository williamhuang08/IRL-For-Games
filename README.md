# Inverse Reinforcement Learning for Games

## Folder Organization

**bc-irl-main**

The bc-irl-main folder contains the project code from Meta Research's [BC-IRL project](https://github.com/facebookresearch/bc-irl). Please see their Github and paper for more detailed information regarding their model and implementation. 

bc-irl-main/.hydra: You can ignore this folder

bc-irl-main/data:
    - checkpoints: If run.py is ran, the reward and policy will be saved in a folder to this location.
    - vids: If run.py is ran, the result of 

## Citation
@article{szot2023bc,
  title={BC-IRL: Learning Generalizable Reward Functions from Demonstrations},
  author={Szot, Andrew and Zhang, Amy and Batra, Dhruv and Kira, Zsolt and Meier, Franziska},
  journal={arXiv preprint arXiv:2303.16194},
  year={2023}
}