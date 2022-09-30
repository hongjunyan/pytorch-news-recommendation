from typing import Callable
import pickle
import argparse
import traceback

from pathlib import Path
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig
from ray import tune
import ray
from mlflow.tracking import MlflowClient
import mlflow


parser = argparse.ArgumentParser()
parser.add_argument('--mind_type', type=str, default="demo", help='One of {demo, small, large}')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--device', type=int, default=-1, help="device index to select")
parser.add_argument('--seed', type=int, default="42", help='random seed')
parser.add_argument('--ray_dir', type=str, default="./ray_tune", help="set a directory to save trained model")
parser.add_argument('--search_method', type=str, default="asha",
                    help="the algorithm/scheduler for searching hyperparameter, "
                         "current support one of {bayesopt, hyperopt, asha, bohb, pbt}")
parser.add_argument('--num_samples', type=int, default=10, help="Maximum number of runs to train.")
parser.add_argument("--gpus_per_trial", type=float, default=0.5, help="train 1/gpu_per_trial trials at once")
args = parser.parse_args()
tracking_client = MlflowClient()
project_dir = str(Path("./").expanduser().absolute())


@mlflow_mixin
def train_a_trial(config):
    with mlflow.start_run(run_name=f"child_run",
                          description="child",
                          nested=True) as child_run:
        mlflow.log_params(config)
        p = mlflow.projects.run(
            uri=project_dir,  # absolute path of current directory
            entry_point="train",
            run_id=child_run.info.run_id,
            parameters={
                "mind_type": args.mind_type,
                "epochs": args.epochs,
                "device": args.device,
                "seed": args.seed,
                "batch_size": config["batch_size"],
                "title_size": config["title_size"],
                "learning_rate": config["learning_rate"],
            },
            experiment_name=config["mlflow"]["experiment_name"],
            synchronous=False,  # Allow the run to fail if a model is not properly created
        )
        succeeded = p.wait()
    if succeeded:
        training_run = tracking_client.get_run(p.run_id)
        metrics = training_run.data.metrics
        best_auc = metrics[f"best_auc"]
    else:
        best_auc = -1

    tune.report(best_auc=best_auc)


def get_tunner(args: argparse.Namespace, trainable_func: Callable, config: dict):
    search_alg = None
    scheduler = None
    if args.search_method == "asha":
        scheduler = ASHAScheduler(
            max_t=args.epochs * 10,  # max iterations(number of tune.report in each trial, in our case is 10)
            grace_period=10,  # patient to stop, each trial must have 10 iterations.
            reduction_factor=2
        )
    if search_alg is None and scheduler is None:
        raise ValueError("Must use search_alg or scheduler to create Tuner")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainable_func),
            resources={"cpu": 0, "gpu": args.gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="best_auc",
            mode="max",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=args.num_samples
        ),
        param_space=config,
        run_config=RunConfig(
            name=config["mlflow"]["experiment_name"],
            local_dir=args.ray_dir
        )
    )

    return tuner


def train():
    ray.shutdown()
    ray.init(dashboard_host="0.0.0.0", dashboard_port=6060)

    with mlflow.start_run() as run:
        # set hyperparameter searching space
        experiment_id = run.info.experiment_id
        experiment_name = tracking_client.get_experiment(experiment_id).name

        config = {
            "batch_size": tune.choice([8, 16, 32]),
            "title_size": tune.choice([10, 30, 50]),
            "learning_rate": tune.choice([0.01, 0.001, 0.0001]),
            "mlflow": {
                "experiment_name": experiment_name,
                "tracking_uri": mlflow.get_tracking_uri()
            }
        }

        tuner = get_tunner(args, train_a_trial, config)

        resutls = tuner.fit()
        best_result = resutls.get_best_result(metric="best_auc", mode="max", scope="all")
        mlflow.set_tag("best params", str(best_result.config))
        mlflow.set_tag("best_auc", str(best_result.metrics['best_auc']))

        # find the best run, log its metrics as the final metrics of this run.
        runs = tracking_client.search_runs([experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'")
        best_auc = best_result.metrics['best_auc']
        for r in runs:
            if r.data.metrics["best_auc"] >= best_auc:
                best_run = r
                best_epoch = r.data.metrics["best_epoch"]
                best_auc = r.data.metrics["best_auc"]

        mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics({
            "best_epoch": best_epoch,
            "best_auc": best_auc,
        })


if __name__ == "__main__":
    train()
