import argparse
import torch
import torch.nn as nn
from models.base import BasicTorchModule
from models.utils import SelfAttention, AttLayer, init_embedding


class NewsEncoder(nn.Module):
    def __init__(self, hparams):
        super(NewsEncoder, self).__init__()
        self.word_embedding = init_embedding(hparams.wordEmb_file)
        self.dropout = nn.Dropout(p=hparams.dropout)
        self.encoder = SelfAttention(hparams.word_emb_dim, hparams.head_num, hparams.head_dim)
        self.attenlayer = AttLayer(int(hparams.head_num * hparams.head_dim), hparams.attention_hidden_dim)

    def forward(self, batch_news_title_indices: torch.IntTensor):
        batch_news_word_emb = self.word_embedding(batch_news_title_indices)  # B x Nh x T x word_emb_dim
        batch_news_word_emb = self.dropout(batch_news_word_emb)  # B x N x T x word_emb_dim
        batch_news_word_repz = self.encoder(
            batch_news_word_emb, batch_news_word_emb, batch_news_word_emb)  # B x N x T x selfatt_dim
        batch_news_word_repz = self.dropout(batch_news_word_repz)  # B x N x T x word_emb_dim
        batch_news_repz = self.attenlayer(batch_news_word_repz)  # B x N x attention_hidden_dim

        return batch_news_repz


class UserEncoder(nn.Module):
    def __init__(self, hparams):
        super(UserEncoder, self).__init__()
        self.encoder = SelfAttention(int(hparams.head_num * hparams.head_dim), hparams.head_num, hparams.head_dim)
        self.attenlayer = AttLayer(int(hparams.head_num * hparams.head_dim), hparams.attention_hidden_dim)

    def forward(self, batch_news_repz):
        batch_news_repz = self.encoder(
            batch_news_repz, batch_news_repz, batch_news_repz)  # B x N x attention_hidden_dim
        batch_user_repz = self.attenlayer(batch_news_repz)  # B x attention_hidden_dim

        return batch_user_repz


class NRMSModel(BasicTorchModule):
    def __init__(self, hparams):
        super(NRMSModel, self).__init__(hparams)
        self.hparams = hparams
        self.news_encoder = NewsEncoder(hparams)
        self.user_encoder = UserEncoder(hparams)

    def forward(self, batch_input: tuple):
        """
        INPUTS:
            batch_input, tuple,
                batch_his_title_indices: torch.IntTensor
                    shape is B x N x T, N is his_size

                batch_cand_title_indices: torch.IntTensor
                    shape is B x N x T, N is candidate size
        """
        batch_his_title_indices, batch_cand_title_indices = batch_input

        # Learn the representations of users from their browsed news
        batch_his_news_repz = self.news_encoder(batch_his_title_indices)  # B x Nh x attention_hidden_dim
        batch_user_repz = self.user_encoder(batch_his_news_repz)  # B x attention_hidden_dim

        # Learn the representation of candidate news from their title
        batch_cand_news_repz = self.news_encoder(batch_cand_title_indices)  # B x Nc x attention_hidden_dim

        # Click prediction
        batch_user_repz = batch_user_repz.unsqueeze(-1)  # B x attention_hidden_dim x 1
        batch_logits = torch.bmm(batch_cand_news_repz, batch_user_repz)  # B x N x 1
        batch_logits = batch_logits.squeeze(dim=-1)  # B x N

        return batch_logits

    def score(self, batch_input: tuple):
        """
        INPUTS:
            batch_input, Tuple
                batch_his_title_indices: torch.IntTensor
                    shape is B x N x T

                batch_one_title_indices: torch.IntTensor
                    shape is B x 1 x T

        RETURN:
            clicked probability of given news title
        """
        batch_logits = self(batch_input)  # B x 1
        batch_y_head = torch.sigmoid(batch_logits)  # B x 1

        return batch_y_head

    def get_input_label_from_iter(self, batch_data):
        """
        DESCRIPTION:
            Helper function for training
        """
        # numpy to tensor
        batch_his_title_indices = torch.from_numpy(batch_data["clicked_title_batch"])
        batch_cand_title_indices = torch.from_numpy(batch_data["candidate_title_batch"])
        batch_labels = torch.from_numpy(batch_data["labels"])
        batch_labels_idx = torch.argmax(batch_labels, dim=1)  # shape is [B]

        if self.hparams.use_gpu:
            batch_his_title_indices = batch_his_title_indices.cuda()
            batch_cand_title_indices = batch_cand_title_indices.cuda()
            batch_labels_idx = batch_labels_idx.cuda()

        return (batch_his_title_indices, batch_cand_title_indices), batch_labels_idx

    def get_news_feature_from_iter(self, batch_data):
        """
        DESCRIPTION:
            Get input of news encoder
        INPUTS:
            batch_data: Dict[str, np.ndarray]
                input batch data from news iterator, dict[feature_name] = feature_value
        RETURN:
            batch_cand_title_indices: torch.FloatTensor
                shape is B x T
        """
        batch_cand_title_indices = torch.from_numpy(batch_data["candidate_title_batch"])
        if self.hparams.use_gpu:
            batch_cand_title_indices = batch_cand_title_indices.cuda()

        return batch_cand_title_indices

    def get_user_feature_from_iter(self, batch_data):
        """
        DESCRIPTION:
            Get input of user encoder
        INPUTS:
            batch_data: Dict[str, np.ndarray]
                input batch data from user iterator, dict[feature_name] = feature_value
        RETURN:
            batch_his_title_indices: torch.FloatTensor
                shape is B x N x T
        """
        batch_his_title_indices = torch.from_numpy(batch_data["clicked_title_batch"])
        if self.hparams.use_gpu:
            batch_his_title_indices = batch_his_title_indices.cuda()

        return batch_his_title_indices
