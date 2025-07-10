from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data.data import BaseData

from utils.rl import ReplayBuffer

class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_ndim: int,
        action_ndim: int,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: Union[nn.Module, str] = "leaky_relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.actor = gnn.MLP(
            in_channels=observation_ndim,
            hidden_channels=n_channels,
            out_channels=action_ndim,
            num_layers=n_layers,
            act=activation,
            dropout=dropout,
            norm=None,
        )

        self.critic = gnn.MLP(
            in_channels=observation_ndim+action_ndim,
            hidden_channels=n_channels,
            out_channels=1,
            num_layers=n_layers,
            act=activation,
            dropout=dropout,
            norm=None,
        )
        self.critic2 = None  # Only used for TD3

class LeagueActorCriticTest(pl.LightningModule):
    def __init__(
        self,
        n_layers: int = 2,
        n_channels: int = 32,
        activation: Union[nn.Module, str] = "leaky_relu",
        dropout: float = 0.0,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        gamma=0.99,
        lam=0.9,
        polyak=0.995,
        buffer_size=10_000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="kwargs")
        self.buffer = ReplayBuffer[BaseData](buffer_size)
        self.automatic_optimization = False

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.polyak = polyak
        self.dropout = dropout

        self.ac = MLPActorCritic(
            25,
            20,
            n_layers,
            n_channels,
            activation,
            dropout,
        )
    
    def training_step(self, batch, batch_idx):
        opt_actor, _ = self.optimizers()
        X, Y = batch
        mu = self.ac.actor.forward(X)
        loss_pi = F.mse_loss(mu, Y)
        opt_actor.zero_grad()
        self.manual_backward(loss_pi)
        opt_actor.step()

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        mu = self.ac.actor.forward(X)
        loss = F.mse_loss(mu, Y)
        return loss

    def test_step(self, batch, batch_idx):
        X, Y = batch
        mu = self.ac.actor.forward(X)
        loss = F.mse_loss(mu, Y)

        return loss

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
