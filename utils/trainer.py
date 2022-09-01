# Standard module
import time
import random

# Third party module
from tqdm import tqdm
import torch
import numpy as np

# Customize module
from utils import HyperParams, groupping_labels, cal_metric, MINDIterator
from models import NPAModel, NRMSModel


class Trainer(object):
    def __init__(self, hparams: HyperParams):
        self.hparams = hparams
        self.support_models = {
            "npa": NPAModel,
            "nrms": NRMSModel
        }

        # Set gpu device if available
        self._set_device()

        # Set seed
        torch.cuda.manual_seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)

        # Create data iterator, directly used MINDIterator from https://github.com/microsoft/recommenders
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
        self.get_input_label_from_iter = self.model.get_input_label_from_iter
        if self.hparams.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

    def evaluate(self, valid_news_file, valid_behaviors_file):
        self.model.eval()
        if self.hparams.support_quick_scoring:
            _, group_labels, group_preds = self._fast_eval(valid_news_file, valid_behaviors_file)
        else:
            _, group_labels, group_preds = self._slow_eval(valid_news_file, valid_behaviors_file)
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res

    def _fast_eval(self, valid_news_file, valid_behaviors_file):
        news_vecs = self.get_news_vec(valid_news_file)  # {nid: ndarray of a news vec}
        user_vecs = self.get_user_vec(valid_news_file, valid_behaviors_file)  # {impr_idx: ndarray of a user vec}
        self.news_vecs = news_vecs
        self.user_vecs = user_vecs

        group_impr_indexes = []
        group_labels = []
        group_preds = []

        valid_iter = self.test_iterator.load_impression_from_file(valid_behaviors_file)
        for (
                impr_index,
                impr_news,
                user_index,
                label,
        ) in tqdm(valid_iter, desc="evaluating"):
            """
            impr_index: int,
            impr_news: List[int],
            user_index: int,
            label: List[int],
            """
            pred = np.dot(np.stack([news_vecs[i] for i in impr_news], axis=0),
                          user_vecs[impr_index])
            group_impr_indexes.append(impr_index)
            group_labels.append(label)
            group_preds.append(pred)

        return group_impr_indexes, group_labels, group_preds

    def _slow_eval(self, valid_news_file, valid_behaviors_file):
        preds = []
        labels = []
        imp_indexes = []
        valid_iter = self.test_iterator.load_data_from_file(valid_news_file, valid_behaviors_file)
        for batch_data in tqdm(valid_iter, desc="evaluating"):
            batch_imp_idx = batch_data["impression_index_batch"]
            batch_labels = batch_data["labels"]
            batch_input, batch_labels_idx = self.get_input_label_from_iter(batch_data)
            batch_y_head = self.model.score(batch_input)

            batch_y_head = batch_y_head.squeeze().cpu().detach().numpy()
            batch_labels = batch_labels.squeeze()
            batch_imp_idx = batch_imp_idx.squeeze()

            preds.extend(batch_y_head)
            labels.extend(batch_labels)
            imp_indexes.extend(batch_imp_idx)

        group_impr_indexes, group_labels, group_preds = groupping_labels(
            labels, preds, imp_indexes
        )

        return group_impr_indexes, group_labels, group_preds

    def get_news_vec(self, valid_news_file):
        news_indices = []
        news_vecs = []
        news_iter = self.test_iterator.load_news_from_file(valid_news_file)
        for batch_data in news_iter:
            batch_news_index = batch_data["news_index_batch"]
            batch_input = self.model.get_news_feature_from_iter(batch_data)  # B x T
            batch_news_vecs = self.model.news_encoder(batch_input)  # B x attention_hidden_dim
            news_indices.extend(batch_news_index)
            news_vecs.extend(batch_news_vecs.cpu().detach().numpy())
        return dict(zip(news_indices, news_vecs))

    def get_user_vec(self, valid_news_file, valid_behaviors_file):
        impr_indices = []
        user_vecs = []
        user_iter = self.test_iterator.load_user_from_file(valid_news_file, valid_behaviors_file)
        for batch_data in user_iter:
            batch_impr_index = batch_data["impr_index_batch"]
            batch_input = self.model.get_user_feature_from_iter(batch_data)  # B x N x T, N is number of clicked history
            batch_his_news_vecs = self.model.news_encoder(batch_input)  # B x N x attention_hidden_dim
            batch_user_vecs = self.model.user_encoder(batch_his_news_vecs)  # B x attention_hidden_dim
            impr_indices.extend(batch_impr_index)
            user_vecs.extend(batch_user_vecs.cpu().detach().numpy())
        return dict(zip(impr_indices, user_vecs))

    def train(self, batch_data):
        self.model.train()
        batch_input, batch_labels_idx = self.get_input_label_from_iter(batch_data)
        batch_logits = self.model(batch_input)

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

        best_eval_auc = 0
        best_eval_res = None
        for epoch in range(1, self.hparams.epochs+1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0

            # Training
            train_start = time.time()
            train_iter = self.train_iterator.load_data_from_file(train_news_file, train_behaviors_file)
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
            eval_res = self.evaluate(valid_news_file, valid_behaviors_file)
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

            # record the best result
            if eval_res["group_auc"] > best_eval_auc:
                best_eval_auc = eval_res["group_auc"]
                best_eval_res = f"Best result at {epoch}: {eval_info}"

        print(best_eval_res)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()
