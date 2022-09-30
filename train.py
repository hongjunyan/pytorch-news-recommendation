import argparse
import traceback
from pathlib import Path
from mlflow.tracking import MlflowClient
import mlflow


from utils import HyperParams, Trainer, download_mind_data

parser = argparse.ArgumentParser()
parser.add_argument('--mind_type', type=str, default="demo", help='One of {demo, small, large}')
parser.add_argument('--model_type', type=str, default="nrms", help='One of {npa, nrms}')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--title_size', type=int, default=10, help='Maximum number of words in title')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--device', type=int, default=-1, help="device index to select")
parser.add_argument('--seed', type=int, default="42", help='random seed')
parser.add_argument('--save_dir', type=str, default="./model_save", help="set a directory to save trained model")
parser.add_argument('--save_best_model', type=bool, default=False, help="save the best model")


# Set hyper-parameter
args = parser.parse_args()
model_config_file = f"config/{args.model_type}.yaml"
hparams = HyperParams(model_config_file)
hparams.update(**args.__dict__)
print("-"*30)
print(hparams)
print("-"*30)


# Download news and user-behaviors data
data_dir = download_mind_data(hparams.mind_type)


# Set the path of news file and user-behaviors data
data_dir = Path(data_dir)
train_news_file = data_dir.joinpath('train', r'news.tsv')
train_behaviors_file = data_dir.joinpath('train', r'behaviors.tsv')
valid_news_file = data_dir.joinpath('valid', r'news.tsv')
valid_behaviors_file = data_dir.joinpath('valid', r'behaviors.tsv')
hparams.wordEmb_file = data_dir.joinpath("utils", "embedding.npy")
hparams.userDict_file = data_dir.joinpath("utils", "uid2index.pkl")
hparams.wordDict_file = data_dir.joinpath("utils", "word_dict.pkl")


# Training
# ref: https://docs.databricks.com/_static/notebooks/mlflow/mlflow-pytorch-training.html
tracking_client = MlflowClient()

with mlflow.start_run() as run:
    try:
        trainer = Trainer(hparams, run.info.run_uuid)
        print(f"Evaluating before training......")
        res = trainer.evaluate(valid_news_file, valid_behaviors_file)
        print(f"Evaluated result before training: {res}")

        print("Start Training ......")
        trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
        if args.save_best_model:
            trainer.save()  # save model in hparams.save_dir
        print("Finish Training")

    except Exception as e:
        tracking_client.set_terminated(run.info.run_uuid, "FAILED")
        mlflow.log_text(str(traceback.format_exc()), "error_log.txt")

    except KeyboardInterrupt as e:
        print(str(traceback.format_exc()))
        tracking_client.set_terminated(run.info.run_uuid, "FAILED")
        mlflow.log_text(f"KeyboardInterrupt: {traceback.format_exc()}", "error_log.txt")
