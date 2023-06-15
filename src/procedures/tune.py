import os
from pathlib import Path

import ray

import torch

from src import procedures
from src import custom_types

from ray.tune.stopper import TrialPlateauStopper

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
RAY_DIR = os.path.join(ROOT_DIR, "ray")


def tune(
    dataset: custom_types.Dataset,
    config,
    name,
    num_cpus=1,
    num_samples=10,
    seed=0,
):
    """Ray Tune.
    :param dataset: dataset class
    :param config: hyperparameter configuration
    :param num_cpus: num_cpus, defaults to 1
    :param num_samples: num_samples, defaults to 1
    :param max_num_epochs: max_num_epochs, defaults to 1
    :param seed: random seed, defaults to 0
    """
    ray.init(num_cpus=num_cpus)
    # Note: Adjust for multiple GPUs
    cuda = int(torch.cuda.is_available())

    verbose = 3
    result = ray.tune.run(
        procedures.train,
        name=name,
        mode="max",
        metric="acc_val_macro",
        resources_per_trial={"cpu": 1, "gpu": cuda},
        config={"cuda": cuda, "seed": seed, "dataset": dataset, **config},
        num_samples=num_samples,
        # scheduler=ray.tune.schedulers.ASHAScheduler(
        #    max_t=int(config["max_num_epochs"]), grace_period=30
        # ),
        # search_alg=ray.tune.search.hyperopt.HyperOptSearch(config), #pip install -U hyperopt
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=["loss_tr", "loss_val", "acc_val_micro", "acc_val_macro"],
            max_report_frequency=30,
        ),
        local_dir=RAY_DIR,
        log_to_file=True,
        verbose=verbose,
        # stop=TrialPlateauStopper(
        #    metric="acc_val_macro", std=0.001, num_results=10, grace_period=30
        # ),
        # checkpoint_score_attr="acc_val_macro",
        # keep_checkpoints_num=5,
    )

    return result
