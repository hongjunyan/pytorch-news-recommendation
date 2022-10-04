from typing import Callable
from tempfile import TemporaryDirectory
import pickle
import argparse
import traceback

from pathlib import Path
from prettytable import PrettyTable
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.schedulers import ASHAScheduler
from ray.air.config import RunConfig
from ray.air import session
from ray import tune
import ray
from mlflow.tracking import MlflowClient
import mlflow.pytorch
import mlflow

from utils import HyperParams, Trainer, download_mind_data


parser = argparse.ArgumentParser()
parser.add_argument('--mind_type', type=str, default="demo", help='One of {demo, small, large}')
parser.add_argument('--model_type', type=str, default="fastformer", help='One of {npa, nrms, fastformer}')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--device', type=int, default=-1, help="device index to select")
parser.add_argument('--seed', type=int, default="42", help='random seed')
parser.add_argument('--save_dir', type=str, default="./model_save", help="set a directory to save trained model")
parser.add_argument('--ray_dir', type=str, default="./ray_tune", help="set a directory to save trained model")
parser.add_argument('--search_method', type=str, default="asha",
                    help="the algorithm/scheduler for searching hyperparameter, "
                         "current support one of {bayesopt, hyperopt, asha, bohb, pbt}")
parser.add_argument('--num_samples', type=int, default=10, help="Maximum number of runs to train.")
parser.add_argument("--gpus_per_trial", type=float, default=0.5, help="train 1/gpu_per_trial trials at once")

# Set hyper-parameter
args = parser.parse_args()
model_config_file = f"config/{args.model_type}.yaml"
hparams = HyperParams(model_config_file)
hparams.update(**args.__dict__)

# Download news and user-behaviors data
data_dir = download_mind_data(args.mind_type)
# Set the path of news file and user-behaviors data
data_dir = Path(data_dir).expanduser().absolute()
train_news_file = data_dir.joinpath('train', r'news.tsv')
train_behaviors_file = data_dir.joinpath('train', r'behaviors.tsv')
valid_news_file = data_dir.joinpath('valid', r'news.tsv')
valid_behaviors_file = data_dir.joinpath('valid', r'behaviors.tsv')
hparams.wordEmb_file = data_dir.joinpath("utils", "embedding.npy")
hparams.userDict_file = data_dir.joinpath("utils", "uid2index.pkl")
hparams.wordDict_file = data_dir.joinpath("utils", "word_dict.pkl")


tracking_client = MlflowClient()
project_dir = str(Path("./").expanduser().absolute())


@mlflow_mixin
def train_a_trial(config: dict):
    trail_id = session.get_trial_id()
    with mlflow.start_run(run_name=trail_id,
                          description="child",
                          nested=True) as child_run:
        mlflow.log_params(config)
        hparams.update(**config)
        trainer = Trainer(hparams)
        trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, is_tune=True)
        # --------Do not write code bellow, because ASHAScheduler will call exit when trainer.fit()-------------


def get_tunner(args: argparse.Namespace, trainable_func: Callable, config: dict):
    search_alg = None
    scheduler = None
    if args.search_method == "asha":
        scheduler = ASHAScheduler(
            max_t=100,  # max iterations (we did tune.report in each iteration)
            grace_period=10,  # patient to stop, each trial must have 10 iterations.
            reduction_factor=3
        )
    if search_alg is None and scheduler is None:
        raise ValueError("Must use search_alg or scheduler to create Tuner")
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainable_func),
            resources={"cpu": 0, "gpu": args.gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="group_auc",
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
    cache = Path("./cache")
    if not cache.exists():
        cache.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run() as run:
        try:
            # Experiment with hyperparameter defined the searching space bellow
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
            results = tuner.fit()

            # Get best result
            resultdf = results.get_dataframe()
            with open(cache.joinpath("results.pkl"), "wb") as f:
                pickle.dump(results, f)
            mlflow.log_artifact(str(cache.joinpath("results.pkl")))
            best_result = results.get_best_result(metric="group_auc", mode="max", scope="all")

            # Save Best Runs
            runs = tracking_client.search_runs([experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'")
            best_auc = 0
            for r in runs:
                # Check status
                run_name = r.data.tags["mlflow.runName"]
                trialdf = resultdf[resultdf["trial_id"]==run_name]
                if trialdf.done.values[0]:
                    tracking_client.set_terminated(r.info.run_id, "FINISHED")

                if r.data.metrics["best_auc"] >= best_auc:
                    best_run = r
                    best_auc = r.data.metrics["best_auc"]
                    best_epoch = r.data.metrics["best_epoch"]
            mlflow.set_tag("best_run", best_run.info.run_id)
            mlflow.set_tag("best_auc_of_trial", best_auc)
            mlflow.set_tag("best_epoch_of_trial", best_epoch)

            # Train final model with best hyperparameter
            with mlflow.start_run(run_name="final_model",
                                  description="child",
                                  nested=True) as child_run:
                best_config = best_result.config
                best_config["epochs"] = int(best_epoch)
                mlflow.log_params(best_config)
                hparams.update(**best_config)
                trainer = Trainer(hparams)
                trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, is_tune=False)
                # Save final result in child run
                with open(cache.joinpath("hparams.pkl"), "wb") as f:
                    pickle.dump(hparams, f)
                mlflow.log_artifact(str(cache.joinpath("hparams.pkl")))
                mlflow.set_tag("best config", best_result.config)
                mlflow.pytorch.log_model(trainer.model, "model")

            # Save error log of trials
            table = PrettyTable(["idx", "error logdir"])
            for idx, logdir in enumerate(resultdf[~resultdf.done].logdir):
                table.add_row([idx, logdir.replace("\\", "/")])
            mlflow.log_text(str(table), "error_trials.txt")

        except Exception as e:
            tracking_client.set_terminated(run.info.run_id, "FAILED")
            mlflow.log_text(str(traceback.format_exc()), "error_log.txt")

        except KeyboardInterrupt as e:
            print(str(traceback.format_exc()))
            tracking_client.set_terminated(run.info.run_id, "FAILED")
            mlflow.log_text(f"KeyboardInterrupt: {traceback.format_exc()}", "error_log.txt")


if __name__ == "__main__":
    train()
