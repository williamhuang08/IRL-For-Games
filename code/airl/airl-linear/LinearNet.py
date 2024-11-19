from imitation.rewards.reward_nets import RewardNet
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn
from stable_baselines3.common import preprocessing

class LinearRewardNet(RewardNet):
    """MLP that takes as input the state, action, next state and done flag.

    These inputs are flattened and then concatenated to one another. Each input
    can enabled or disabled by the `use_*` constructor keyword arguments.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        **kwargs,
    ):
        """Builds reward MLP.

        Args:
            observation_space: The observation space.
            action_space: The action space.
            use_state: should the current state be included as an input to the MLP?
            use_action: should the current action be included as an input to the MLP?
            use_next_state: should the next state be included as an input to the MLP?
            use_done: should the "done" flag be included as an input to the MLP?
            kwargs: passed straight through to `build_mlp`.
        """
        super().__init__(observation_space, action_space)
        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        # full_build_mlp_kwargs: Dict[str, Any] = {
        #     "hid_sizes": (32, 32),
        #     **kwargs,
        #     # we do not want the values below to be overridden
        #     "in_size": combined_size,
        #     "out_size": 1,
        #     "squeeze_output": True,
        # }

        self.linear = nn.Linear(combined_size, 1)  # Single output for reward



    def forward(self, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor) -> th.Tensor:
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        # Concatenate inputs
        inputs_concat = th.cat(inputs, dim=1)

        # Compute the linear output
        outputs = self.linear(inputs_concat)
        
        # Ensure output shape matches expected shape
        outputs = outputs.view(-1, 1)  # Reshape to (batch_size, 1)
        
        return outputs