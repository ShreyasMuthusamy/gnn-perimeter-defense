from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch_geometric.data.data import BaseData

from architecture.gnn import GCN

class GNNActor(nn.Module):
    def __init__(
        self,
        state_ndim: int,
        action_ndim: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_ndim = state_ndim
        self.action_ndim = action_ndim

        self.gnn = GCN(
            state_ndim,
            action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

        self.log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_ndim)))

    def forward(self, state: torch.Tensor, data: BaseData):
        action = self.gnn.forward(state, data.edge_index, data.edge_attr)
        mu = action[:, : self.action_ndim]
        sigma = torch.exp(self.log_std)
        return mu, sigma

    def distribution(self, state: torch.Tensor, data: BaseData) -> Normal:
        mu, sigma = self.forward(state, data)
        return Normal(mu, sigma)

    def log_prob(self, pi: Normal, action: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(action)

    def policy(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        """
        Sample from a Gaussian distribution.

        Args:
            mu: (batch_size, N) mean of the Gaussian distribution
            sigma: (batch_size, N) standard deviation of the Gaussian distribution.
            action: (batch_size, N) Optional action. If given returns tuple action, log_prob, entropy.

        Returns:
            action: (batch_size, N) sample action or given action.
            log_prob: (batch_size, 1) log probability of the action (if action is given).
            entropy: (batch_size, 1) entropy of the action (if action is given).
        """
        if action is None:
            eps = torch.randn_like(mu)
            action = torch.tanh(mu + sigma * eps)
            assert isinstance(action, torch.Tensor)
            return torch.tanh(action)

        dist = Normal(mu, sigma)
        return dist, self.log_prob(dist, action)


class GNNCritic(nn.Module):
    def __init__(
        self,
        state_ndim: int,
        action_ndim: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gnn = GCN(
            state_ndim + action_ndim,
            1,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        data: BaseData,
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        y = self.gnn.forward(x, data.edge_index, data.edge_attr)
        return y.squeeze(-1)

class GNNActorCritic(nn.Module):
    def __init__(
        self,
        observation_ndim: int,
        action_ndim: int,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.actor = GNNActor(
            observation_ndim,
            action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

        self.critic = GNNCritic(
            observation_ndim,
            action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )
        self.critic2 = None  # Only used for TD3

    def policy(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        return self.actor.policy(mu, sigma, action)

    def step(self, state: torch.Tensor, data: BaseData):
        with torch.no_grad():
            pi = self.actor.distribution(state, data)
            action = pi.sample()
            logp = self.actor.log_prob(pi, action)
            value = self.value(state, data)

        return action, logp, value

    def action(self, state: torch.Tensor, data: BaseData):
        with torch.no_grad():
            return self.actor(state, data)[0].cpu().numpy()
