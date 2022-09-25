import os
import argparse
from utils import HyperParams, Trainer, download_mind_data


parser = argparse.ArgumentParser()
parser.add_argument('--mind_type', type=str, default="demo", help='One of {demo, small, large}')
parser.add_argument('--model_type', type=str, default="nrms", help='One of {npa, nrms}')
parser.add_argument('--batch_size', type=int, default="32", help='batch size')
parser.add_argument('--epochs', type=int, default="1", help='number of epochs')
parser.add_argument('--title_size', type=int, default="10", help='Maximum number of words in title')
parser.add_argument('--device', type=int, default=None, help="device index to select")
parser.add_argument('--seed', type=int, default="42", help='random seed')
parser.add_argument('--save_dir', type=str, default="./model_save", help="set a directory to save trained model")

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
train_news_file = os.path.join(data_dir, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_dir, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_dir, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_dir, 'valid', r'behaviors.tsv')
hparams.wordEmb_file = os.path.join(data_dir, "utils", "embedding.npy")
hparams.userDict_file = os.path.join(data_dir, "utils", "uid2index.pkl")
hparams.wordDict_file = os.path.join(data_dir, "utils", "word_dict.pkl")


# Training
trainer = Trainer(hparams)
print(f"Evaluating before training......")
res = trainer.evaluate(valid_news_file, valid_behaviors_file)
print(f"Evaluated result before training: {res}")

print("Start Training ......")
trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
trainer.save()  # save model in hparams.save_dir
print("Finish Training")