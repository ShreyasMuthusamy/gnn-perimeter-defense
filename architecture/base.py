from typing import Iterator, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from game.game import LeagueEnv
from utils.rl import ExperienceSourceDataset
from architecture.rl import GNNActorCritic

class LeagueActorCritic(pl.LightningModule):
    def __init__(
        self,
        n_taps: int = 4,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: Union[nn.Module, str] = "leaky_relu",
        mlp_read_layers: int = 1,
        mlp_per_gnn_layers: int = 0,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.0,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        gamma=0.99,
        lam=0.9,
        polyak=0.995,
        max_steps=200,
        n_agents: int = 100,
        width: float = 10.0,
        agent_radius: float = 0.05,
        agent_margin: float = 0.05,
        scenario: str = "uniform",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="kwargs")

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.polyak = polyak
        self.max_steps = max_steps
        self.dropout = dropout
        self.agent_radius = agent_radius
        self.agent_margin = agent_margin

        self.env = LeagueEnv(
            n_agents=n_agents,
            width=width,
            scenario=scenario,
            agent_radius=agent_radius + agent_margin,
        )

        self.ac = GNNActorCritic(
            self.env.observation_ndim,
            self.env.action_ndim,
            n_taps,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def clip_action(self, action):
        magnitude = torch.norm(action, dim=-1)
        magnitude = torch.clip(magnitude, 0, self.env.max_vel)
        tmp = action[:, 0] + 1j * action[:, 1]  # Assumes two dimensions
        angles = torch.angle(tmp)
        action_x = (magnitude * torch.cos(angles))[:, None]
        action_y = (magnitude * torch.sin(angles))[:, None]
        return torch.cat([action_x, action_y], dim=1)

    def rollout_start(self):
        """
        Called before rollout starts.
        """
        return None

    def rollout_step(self, data: BaseData):
        """
        Called after rollout step.
        """
        data.action = self.ac.policy(data.mu, data.sigma)
        next_state, reward, done, _ = self.env.step(
            data.action.detach().cpu().numpy()  # type: ignore
        )
        coverage = self.env.coverage()
        n_collisions = self.env.n_collisions(r=self.agent_radius)
        return data, next_state, reward, done, coverage, n_collisions

    @torch.no_grad()
    def rollout(self, render=False) -> tuple[List[BaseData], List[np.ndarray]]:
        self.rollout_start()
        episode = []
        observation, centralized_state = self.env.reset()
        data = self.to_data(observation, centralized_state, 0, self.env.adjacency())
        frames = []
        for step in range(self.max_steps):
            if render:
                frames.append(self.env.render(mode="rgb_array"))

            # sample action
            data.mu, data.sigma = self.ac.actor(data.state, data)

            # take step
            (
                data,
                next_state,
                reward,
                done,
                coverage,
                n_collisions,
            ) = self.rollout_step(data)

            # add additional attributes
            next_data = self.to_data(
                next_state, step + 1, self.env.adjacency()
            )
            data.reward = torch.as_tensor(reward).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.coverage = torch.tensor([coverage]).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.n_collisions = torch.tensor([n_collisions]).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.next_state = next_data.state
            data.done = torch.tensor(done, dtype=torch.bool, device=self.device)  # type: ignore

            episode.append(data)
            data = next_data
            if done:
                break

        return episode, frames

    def critic_loss(
        self,
        q: torch.Tensor,
        qprime: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(q, reward + torch.logical_not(done) * self.gamma * qprime)

    def optimizers(self):
        opts = super().optimizers()
        if not isinstance(opts, list):
            raise ValueError(
                "Expected a list of optimizers: an actor and multiple critics. Double check that `configure_optimizers` returns multiple optimziers."
            )
        return opts

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.ac.actor.parameters(),
                lr=self.actor_lr,
                weight_decay=self.weight_decay,
            ),
            torch.optim.AdamW(
                self.ac.critic.parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        ]

    def to_data(self, state, step, adjacency) -> BaseData:
        if isinstance(adjacency, list):
            data = []
            for i, adj in enumerate(adjacency):
                data.append(self.to_data(state[i], step, adj))
            return Batch.from_data_list(data)
        step = step / self.max_steps
        step = np.tile(step, state.shape[0])[:, None]
        state = np.concatenate([state, step], axis=1)
        state = torch.from_numpy(state).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        centralized_state = torch.from_numpy(centralized_state).to(
            dtype=self.dtype, device=self.device
        )
        # assert state.shape == (self.env.n_nodes, self.env.observation_ndim)
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)  # type: ignore
        return Data(
            state=state,
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=state.shape[0],
        )

    def batch_generator(self, *args, **kwargs) -> Iterator:
        """
        Generate batches of data.

        Args:
            n_episodes: Number of new episodes to generate. Can be zero.
            render: Whether to render the environment as we generate samples.
            use_buffer: Whether to use a replay buffer to generate samples.
        """
        raise NotImplementedError("Should be overriden by subclasses.")

    def _dataloader(self, **kwargs):
        return DataLoader(
            ExperienceSourceDataset(self.batch_generator, **kwargs),  # type: ignore
            batch_size=self.batch_size,
        )
