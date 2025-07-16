import argparse
import json
import os
import sys
import typing
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from scipy.spatial.distance import cdist
from tqdm import tqdm
from wandb.wandb_run import Run

from lib.lightning import (
    LeagueActorCritic,
    LeagueImitation,
)


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()

    # program argument2
    parser.add_argument(
        "operation",
        type=str,
        default="td3",
        choices=[
            "imitation",
        ],
        help="The operation to perform.",
    )
    operation = sys.argv[1]

    # common args
    group = parser.add_argument_group("Simulation")
    group.add_argument("--n_trials", type=int, default=10)

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    if operation in ("imitation",):
        # training arguments
        training_group = parser.add_argument_group("Training")
        training_group.add_argument("--no_log", action="store_false", dest="log")
        training_group.add_argument("--test", action="store_true")
        training_group.add_argument("--max_epochs", type=int, default=100)
        training_group.add_argument("--patience", type=int, default=10)
        training_group.add_argument("--notes", type=str, default="")

    params = parser.parse_args()
    if params.operation == "imitation":
        imitation(params)
    else:
        raise ValueError(f"Invalid operation {params.operation}.")


def load_model(uri: str, best: bool = True) -> tuple[LeagueActorCritic, str]:
    """Load a model from a uri.

    Args:
        uri (str): The uri of the model to load. By default this is a path to a file. If you want to use a wandb model, use the format wandb://<user>/<project>/<run_id>.
        cls: The class of the model to load.
    """
    with TemporaryDirectory() as tmpdir:
        if uri.startswith("wandb://"):
            import wandb

            user, project, run_id = uri[len("wandb://") :].split("/")
            suffix = "best" if best else "latest"

            # Download the model from wandb to temporary directory
            api = wandb.Api()
            artifact = api.artifact(
                f"{user}/{project}/model-{run_id}:{suffix}", type="model"
            )
            artifact.download(root=tmpdir)
            uri = f"{tmpdir}/model.ckpt"
            # set the name and model_str
            name = run_id
            model_str = api.run(f"{user}/{project}/{run_id}").config["operation"]
        else:
            name = os.path.basename(uri).split(".")[0]
            if "imitation" in uri:
                model_str = "imitation"
            else:
                raise ValueError(f"Invalid model uri {uri}.")

        cls = get_model_cls(model_str)
        model = cls.load_from_checkpoint(uri)
        return model, name


def make_trainer(params):
    logger = False
    callbacks: list[pl.Callback] = [
        EarlyStopping(
            monitor="val/reward",
            mode="max",
            patience=params.patience,
        ),
    ]

    if params.log:
        logger = WandbLogger(
            project="graph-games",
            save_dir="logs",
            config=params,
            log_model=True,
            notes=params.notes,
        )
        logger.log_hyperparams(params)
        run = typing.cast(Run, logger.experiment)
        run.log_code(
            Path(__file__).parent.parent,
            include_fn=lambda path: (
                path.endswith(".py")
                and "logs" not in path
                and ("src" in path or "scripts" in path)
            ),
        )
        callbacks += [
            ModelCheckpoint(
                monitor="val/mu_loss",
                mode="min",
                dirpath=f"logs/{params.operation}/{run.id}/",
                filename="best",
                auto_insert_metric_name=False,
                save_last=True,
                save_top_k=1,
            )
        ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=params.log,
        precision=32,
        max_epochs=params.max_epochs,
        default_root_dir="logs/",
        check_val_every_n_epoch=1,
    )
    return trainer


def find_checkpoint(operation):
    candidates = [
        f"./{operation}/checkpoints/best.ckpt",
        f"./{operation}/checkpoints/last.ckpt",
    ]
    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt


def imitation(params):
    trainer = make_trainer(params)
    model = LeagueImitation(**vars(params))
    ckpt_path = "./imitation/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params.test:
        trainer.test(model)


def get_model_cls(model_str) -> typing.Type[LeagueActorCritic]:
    if model_str == "imitation":
        return LeagueImitation
    raise ValueError(f"Invalid model {model_str}.")


if __name__ == "__main__":
    main()
