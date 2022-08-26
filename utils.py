# standard module
import os
from typing import Union
import random
import json
import time
# third-party module
from tqdm import tqdm
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.deeprec.deeprec_utils import cal_metric
import torch
import numpy as np
from pathlib import Path
# customize module
from models import NPAModel


class HyperParams(object):
    def __init__(self, cfg_file: Union[Path, str]):
        cfg_file = Path(cfg_file)
        if cfg_file.exists():
            with open(cfg_file, "r") as f:
                cfg = json.load(f)

            self._cfg = cfg
            for hparam in cfg:
                setattr(self, hparam, cfg[hparam])

    def update(self, **kargs):
        self._cfg.update(kargs)
        for hparam in kargs:
            setattr(self, hparam, kargs[hparam])

    def __repr__(self):
        return f"Hyper-Parameter: {self._cfg}"


def download_mind_data(mind_type: str) -> str:
    """
    INPUT:
        mind_type: str,
            one of {demo, small, large}
    """
    data_dir = f"./MIND_data_{mind_type}"
    train_news_file = os.path.join(data_dir, 'train', r'news.tsv')
    valid_news_file = os.path.join(data_dir, 'valid', r'news.tsv')
    yaml_file = os.path.join(data_dir, "utils", r'npa.yaml')

    mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(mind_type)

    if not os.path.exists(train_news_file):
        download_deeprec_resources(mind_url, os.path.join(data_dir, 'train'), mind_train_dataset)

    if not os.path.exists(valid_news_file):
        download_deeprec_resources(mind_url, os.path.join(data_dir, 'valid'), mind_dev_dataset)
    if not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.z20.web.core.windows.net/newsrec/',
                                   os.path.join(data_dir, 'utils'), mind_utils)

    return data_dir


def groupping_labels(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.
    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.
    Returns:
        list, list, list:
        - Keys after group.
        - Labels after group.
        - Preds after group.
    """

    all_keys = list(set(group_keys))
    all_keys.sort()
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for label, p, k in zip(labels, preds, group_keys):
        group_labels[k].append(label)
        group_preds[k].append(p)

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return all_keys, all_labels, all_preds


class Trainer(object):
    def __init__(self, hparams: HyperParams):
        self.hparams = hparams
        self.support_models = {
            "npa": NPAModel
        }

        # Set gpu device if available
        self._set_device()

        # Set seed
        torch.cuda.manual_seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)

        # Create data iterator, directly used MINDIterator from https://github.com/microsoft/recommenders
        # If not found MINDIterator, please run command `pip install recommenders` first.
        self.train_iterator = MINDIterator(hparams, hparams.npratio, col_spliter="\t")
        self.test_iterator = MINDIterator(hparams, col_spliter="\t")

        # Build graph
        self._build_graph()

        # Show model information
        print(f"Total Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        print(f"Number of Users: {len(self.train_iterator.uid2index)}, Number of vocab: {len(self.train_iterator.word_dict)}")

    def _set_device(self):
        # Set device
        msg = ""
        self.hparams.use_gpu = False
        if torch.cuda.is_available():
            if self.hparams.device is None:
                print("WARNING: You have a CUDA device, should run with --device 0")
                msg = "use cpu"
            else:
                torch.cuda.set_device(self.hparams.device)
                self.hparams.use_gpu = True
                msg = f"use gpu {self.hparams.device}"
        else:
            msg = "use cpu, because cuda is not available"
        print(f"Model {msg}")

    def _build_graph(self):
        self.hparams.user_num = len(self.train_iterator.uid2index)
        model_class = self.support_models[self.hparams.model_type]
        self.model = model_class(self.hparams)
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

    def _get_input_label_from_iter(self, batch_data):
        # numpy to tensor
        batch_user_indices = torch.from_numpy(batch_data["user_index_batch"]).squeeze()
        batch_his_title_indices = torch.from_numpy(batch_data["clicked_title_batch"])
        batch_one_title_indices = torch.from_numpy(batch_data["candidate_title_batch"])
        batch_labels = torch.from_numpy(batch_data["labels"])
        batch_labels_idx = torch.argmax(batch_labels, dim=1)  # shape is [B]

        if self.hparams.use_gpu:
            batch_user_indices = batch_user_indices.cuda()
            batch_his_title_indices = batch_his_title_indices.cuda()
            batch_one_title_indices = batch_one_title_indices.cuda()
            batch_labels_idx = batch_labels_idx.cuda()

        return (batch_user_indices, batch_his_title_indices, batch_one_title_indices), batch_labels_idx

    def get_train_iter(self, train_news_file, train_behaviors_file):
        """
        Directly used MINDIterator (ref: https://github.com/microsoft/recommenders).
        If not found MINDIterator, please run command `pip install recommenders` first.
        """
        train_iter = self.train_iterator.load_data_from_file(train_news_file, train_behaviors_file)
        return train_iter

    def get_valid_iter(self, valid_news_file, valid_behaviors_file):
        valid_iter = self.test_iterator.load_data_from_file(valid_news_file, valid_behaviors_file)
        return valid_iter

    def evaluate(self, valid_iter):
        preds = []
        labels = []
        imp_indexes = []
        self.model.eval()
        for batch_data in tqdm(valid_iter, desc="evaluating"):
            batch_imp_idx = batch_data["impression_index_batch"]
            batch_labels = batch_data["labels"]
            batch_input_data, batch_labels_idx = self._get_input_label_from_iter(batch_data)
            batch_user_indices, batch_his_title_indices, batch_one_title_indices = batch_input_data

            batch_y_head = self.model.score(
                batch_user_indices,
                batch_his_title_indices,
                batch_one_title_indices
            )

            batch_y_head = batch_y_head.squeeze().cpu().detach().numpy()
            batch_labels = batch_labels.squeeze()
            batch_imp_idx = batch_imp_idx.squeeze()

            preds.extend(batch_y_head)
            labels.extend(batch_labels)
            imp_indexes.extend(batch_imp_idx)

        group_impr_indexes, group_labels, group_preds = groupping_labels(
            labels, preds, imp_indexes
        )

        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        self.model.train()
        return res

    def train(self, batch_data):
        batch_input_data, batch_labels_idx = self._get_input_label_from_iter(batch_data)
        batch_user_indices, batch_his_title_indices, batch_cand_title_indices = batch_input_data
        batch_logits = self.model(
            batch_user_indices,
            batch_his_title_indices,
            batch_cand_title_indices
        )

        batch_log_y_head = torch.log_softmax(batch_logits, dim=-1)  # B x N
        loss = self.criterion(batch_log_y_head, batch_labels_idx)  # default mean of nll loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def fit(self,
            train_news_file,
            train_behaviors_file,
            valid_news_file,
            valid_behaviors_file):

        for epoch in range(1, self.hparams.epochs+1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0

            # Training
            train_start = time.time()
            train_iter = self.get_train_iter(train_news_file, train_behaviors_file)
            tqdm_util = tqdm(train_iter, desc="training")
            for batch_data in tqdm_util:
                step_data_loss = self.train(batch_data)
                epoch_loss += step_data_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    tqdm_util.set_description(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, epoch_loss / step, step_data_loss
                        )
                    )
            train_end = time.time()
            train_time = train_end - train_start
            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("negative log likelihood loss", epoch_loss / step)]
                ]
            )

            # Evaluation
            eval_start = time.time()
            valid_iter = self.get_valid_iter(valid_news_file, valid_behaviors_file)
            eval_res = self.evaluate(valid_iter)
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            print(
                "at epoch {0:d}".format(epoch)
                + "\ntrain info: "
                + train_info
                + "\neval info: "
                + eval_info
            )

            print(
                "at epoch {0:d} , train time: {1:.1f}s eval time: {2:.1f}s".format(
                    epoch, train_time, eval_time
                )
            )

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()
