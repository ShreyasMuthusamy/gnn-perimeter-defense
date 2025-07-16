from typing import Iterator, List, Union
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix

import lib.core.time_logger as TLOG
from lib.utils.rl import ExperienceSourceDataset
from architecture.rl import GNNActorCritic
from game import run_game

class LeagueActorCritic(pl.LightningModule):
    def __init__(
        self,
        field_size: int = 10,
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
        agent_radius: float = 0.05,
        agent_margin: float = 0.05,
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

        self.ac = GNNActorCritic(
            3,      # number of input channels
            1,      # number of output channels
            field_size,
            n_layers,
            n_channels,
            activation,
            mlp_read_layers,
            mlp_per_gnn_layers,
            mlp_hidden_channels,
            dropout,
        )

    def rollout_start(self):
        """
        Called before rollout starts.
        """
        return None

    @torch.no_grad()
    def rollout(self) -> List[BaseData]:
        current_path = Path(__file__).resolve()
        root_path = current_path.parent

        self.rollout_start()
        
        import policies.attacker.V1R1_MSU_Atk as attacker
        import policies.defender.V1R1_MSU_Def as defender
        
        logger = TLOG.TimeLogger('rollout')
        G, _, _, _, _ = run_game(
            "F5A10D10_0ac5d2_r02.yml",
            root_dir=str(root_path),
            attacker_strategy=attacker,
            defender_strategy=defender,
            logger=logger
        )

        return logger.to_data(G, dtype=self.dtype, device=self.device)

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
