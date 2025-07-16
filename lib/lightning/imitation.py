import inspect

import torch.nn.functional as F
from torch_geometric.data.data import BaseData

from lib.lightning.base import LeagueActorCritic
from lib.utils.rl import ReplayBuffer


class LeagueImitation(LeagueActorCritic):
    def __init__(
        self,
        buffer_size: int = 100_000,
        **kwargs,
    ):
        """
        Args:
            buffer_size: size of the replay buffer
            target_policy: the target policy to use for the expert
            expert_probability: probability of sampling from the expert
            render: whether to render the environment
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.buffer = ReplayBuffer[BaseData](buffer_size)
        self.automatic_optimization = False

    def training_step(self, data: BaseData, batch_idx):
        opt_actor, opt_critic = self.optimizers()

        # actor step
        mu, _ = self.ac.actor.forward(data.state, data)
        loss_pi = F.mse_loss(mu, data.action)
        self.log("train/mu_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        opt_actor.zero_grad()
        self.manual_backward(loss_pi)
        opt_actor.step()

    def validation_step(self, data: BaseData, batch_idx):
        mu, _ = self.ac.actor.forward(data.state, data)
        loss = F.mse_loss(mu, data.action)
        self.log("val/mu_loss", loss, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "val/payoff", data.payoff.mean(), prog_bar=True, batch_size=data.batch_size
        )

        return loss

    def test_step(self, data: BaseData, batch_idx):
        mu, _ = self.ac.actor.forward(data.state, data)
        loss = F.mse_loss(mu, data.action)
        self.log("test/mu_loss", loss, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "test/payoff", data.payoff.mean(), prog_bar=True, batch_size=data.batch_size
        )

        return loss

    def batch_generator(
        self, n_episodes=1, use_buffer=True, training=True
    ):
        # set model to appropriate mode
        self.train(training)

        data = []
        for _ in range(n_episodes):
            episode = self.rollout()
            data.extend(episode)
        if use_buffer:
            self.buffer.extend(data)
            data = self.buffer.collect(shuffle=True)
        return iter(data)

    def train_dataloader(self):
        return self._dataloader(
            n_episodes=10, use_buffer=True, training=True
        )

    def val_dataloader(self):
        return self._dataloader(
            n_episodes=1, use_buffer=False, training=False
        )

    def test_dataloader(self):
        return self._dataloader(
            n_episodes=10, use_buffer=False, training=False
        )