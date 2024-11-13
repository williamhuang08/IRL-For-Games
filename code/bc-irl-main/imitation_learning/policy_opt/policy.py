# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from rl_utils.models import (FixedCategorical, FixedNormal, SimpleCNN,
                             build_rnn_state_encoder)
from tensordict.nn import (TensorDictModule, TensorDictSequential)

from torchrl.modules import (ActorValueOperator, ConvNet, ProbabilisticActor,
                             ValueOperator)


def init_weights(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        self.apply(partial(init_weights, gain=0.01))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, std_init, squash_mean):
        super().__init__()

        if squash_mean:
            self.fc_mean = nn.Sequential(
                nn.Linear(num_inputs, num_outputs),
                nn.Tanh(),
            )
        else:
            self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = nn.Parameter(torch.full((1, num_outputs), float(std_init)))
        self.apply(init_weights)

    def forward(self, x):
        action_mean = self.fc_mean(x)

        action_logstd = self.logstd.expand_as(action_mean)
        return FixedNormal(action_mean, action_logstd.exp())

class DiagGaussianParams(nn.Module):
    """Returns the parameters for a normal distribution."""

    def __init__(self, hidden_size, action_dim, std_init, squash_mean):
        super().__init__()

        if squash_mean:
            self.fc_mean = nn.Sequential(
                nn.Linear(hidden_size, action_dim),
                nn.Tanh(),
            )
        else:
            self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.logstd = nn.Parameter(torch.full((1, action_dim), float(std_init)))
        self.apply(init_weights)

    def forward(self, x):
        action_mean = self.fc_mean(x)

        action_logstd = self.logstd.expand_as(action_mean)
        return action_mean, action_logstd.exp()

class Policy(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        action_is_discrete,
        hidden_size,
        recurrent_hidden_size,
        is_recurrent,
        std_init=0.0,
        squash_mean=False,
    ):
        super().__init__()
        if isinstance(obs_shape, dict):
            is_visual_obs = any([len(v) == 3 for k, v in obs_shape.items()])
        else:
            is_visual_obs = len(obs_shape) == 3

        if is_visual_obs:
            self.backbone = SimpleCNN(obs_shape, hidden_size)
            input_size = hidden_size
        else:
            self.backbone = nn.Sequential()
            input_size = obs_shape[0]

        if is_recurrent:
            self.rnn_encoder = build_rnn_state_encoder(
                recurrent_hidden_size, recurrent_hidden_size
            )
        else:
            # Pass through
            self.rnn_encoder = lambda hidden, hxs, _: (hidden, hxs)

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.apply(partial(init_weights, gain=np.sqrt(2)))

        if action_is_discrete:
            self.actor_dist = Categorical(hidden_size, action_dim)
        else:
            self.actor_dist = DiagGaussian(
                hidden_size, action_dim, std_init, squash_mean
            )

    def get_value(self, obs, hxs, masks):
        hidden, _ = self.forward(obs, hxs, masks)
        return self.critic(hidden)

    def evaluate_actions(self, obs, hxs, masks, action):
        hidden, hxs = self.forward(obs, hxs, masks)
        critic_value = self.critic(hidden)

        actor_hidden = self.actor(hidden)
        dist = self.actor_dist(actor_hidden)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return {
            "log_prob": action_log_probs,
            "value": critic_value,
            "dist_entropy": dist_entropy,
        }

    def forward(self, obs, hxs, masks):
        hidden = self.backbone(obs)
        hidden, hxs = self.rnn_encoder(hidden, hxs, masks)
        return hidden, hxs

    def get_action_dist(self, obs, hxs, masks):
        hidden, hxs = self.forward(obs, hxs, masks)
        actor_hidden = self.actor(hidden)
        return self.actor_dist(actor_hidden)

    def act(self, obs, hxs, masks, is_eval=False):
        hidden, hxs = self.forward(obs, hxs, masks)

        critic_value = self.critic(hidden)

        actor_hidden = self.actor(hidden)
        dist = self.actor_dist(actor_hidden)
        if is_eval:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy()

        return {
            "actions": action,
            "action_log_probs": action_log_probs,
            "value_preds": critic_value,
            "hxs": hxs,
            "extra": {
                "dist_entropy": dist_entropy,
            },
        }


# class Policy(nn.Module):
#     def __init__(
#         self,
#         obs_shape,
#         action_dim,
#         action_is_discrete,
#         hidden_size,
#         recurrent_hidden_size,
#         is_recurrent,
#         num_envs,
#         std_init=0.0,
#         squash_mean=False,
#     ):
#         super(Policy, self).__init__()

#         self.num_envs = num_envs
#         self.hidden_size = hidden_size
#         self.recurrent_hidden_size = recurrent_hidden_size
#         self.action_is_discrete = action_is_discrete

#         # If the observation is discrete (state index), use a simple FC network
#         if isinstance(obs_shape, tuple) and len(obs_shape) > 1:
#             input_size = obs_shape[0]  # For vectorized state representation [view size]
#         else:
#             input_size = obs_shape  # For discrete state representation

#         # Define the observation processing part (fully connected layers)
#         self.backbone = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.Tanh(),
#         )

#         # Recurrent layer (optional)
#         if is_recurrent:
#             self.rnn_encoder = nn.LSTM(hidden_size, recurrent_hidden_size)
#         else:
#             self.rnn_encoder = nn.Identity()  # No recurrent layer if not needed

#         # Actor network for choosing actions (policy)
#         actor = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, action_dim),  # Output is the size of action space
#         )

#         # Critic network for value estimation
#         critic = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.Tanh(),
#             nn.Linear(hidden_size, 1),  # Output is a scalar (state value)
#         )

#         # Initialize the networks
#         actor.apply(self.init_weights)
#         critic.apply(self.init_weights)

#         # For discrete action space, use categorical distribution
#         if self.action_is_discrete:
#             dist_params_net = nn.Linear(hidden_size, action_dim)
#             distribution_class = Categorical
#         else:
#             dist_params_net = nn.Linear(hidden_size, action_dim)
#             distribution_class = FixedNormal

#         actor.append(dist_params_net)

#         # Create TensorDictModule for the actor and critic
#         self.actor_module = TensorDictModule(module=actor, in_keys=["hidden"], out_keys=["logits"])
#         self.critic_module = TensorDictModule(module=critic, in_keys=["hidden"], out_keys=["state_value"])

#         self.hidden = nn.Sequential(self.backbone, self.rnn_encoder)
#         self.module = ActorValueOperator(self.hidden, self.actor_module, self.critic_module)

#         self.actor = self.module.get_policy_operator()
#         self.critic = self.module.get_value_operator()

#     def init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.zeros_(m.bias)

#     def get_value(self, td):
#         return self.critic(td)["state_value"]

#     def evaluate_actions(self, td):
#         critic_td = self.critic(td)
#         critic_value = critic_td["state_value"]
#         action = td["action"]

#         dist = self.actor.get_dist(td)
#         action_log_prob = dist.log_prob(action)
#         dist_entropy = dist.entropy()

#         return {
#             "log_prob": action_log_prob,
#             "value": critic_value,
#             "dist_entropy": dist_entropy,
#         }

#     def forward(self, td):
#         return self.hidden(td)

#     def get_action_dist(self, td):
#         self.forward(td)
#         dist = self.actor.get_dist(td)
#         return dist

#     def act(self, td, deterministic=False):
#         hidden_td = self.forward(td)
#         dist = self.actor.get_dist(td)

#         if deterministic:
#             action = dist.mode
#         else:
#             action = dist.sample()

#         action_log_probs = dist.log_prob(action)
#         dist_entropy = dist.entropy()

#         td["action"] = action
#         td["sample_log_prob"] = action_log_probs
#         td["dist_entropy"] = dist_entropy
#         return td
