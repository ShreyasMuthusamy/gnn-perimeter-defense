import os, sys

import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from architecture.test import LeagueActorCriticTest
from utils.dataset import get_data

def main():
    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        logger=False,
        devices=1,
        enable_checkpointing=False,
        precision=32,
        max_epochs=3,
        default_root_dir="logs/",
        check_val_every_n_epoch=1,
    )

    model = LeagueActorCriticTest()
    data = get_data('graph_a')
    X, Y = data[:, :25], data[:, 25:]
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)
    set = TensorDataset(X, Y)
    loader = DataLoader(set)

    trainer.fit(model, train_dataloaders=loader)
    torch.save(model, './models/test.pt')

if __name__ == '__main__':
    main()
