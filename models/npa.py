import argparse
import torch
import torch.nn as nn
from models.base import BasicTorchModule
from models.news_encoder import NewsEncoder
from models.utils import PersonalizedAttentivePooling


class NPAModel(BasicTorchModule):
    def __init__(self, hparams):
        super(NPAModel, self).__init__(hparams)
        self.hparams = hparams
        # the first dimension of user_embedding is always 0's. (for unknown user)
        self.user_embedding = nn.Embedding(hparams.user_num+1, hparams.user_emb_dim, padding_idx=0)
        self.linear_q_word = nn.Linear(hparams.user_emb_dim, hparams.attention_hidden_dim)
        self.linear_q_news = nn.Linear(hparams.user_emb_dim, hparams.attention_hidden_dim)
        self.news_encoder = NewsEncoder(hparams)
        self.news_attention_pool = PersonalizedAttentivePooling(
            hparams.filter_num,
            hparams.attention_hidden_dim
        )
        nn.init.zeros_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.linear_q_word.weight)
        nn.init.zeros_(self.linear_q_word.bias)
        nn.init.xavier_normal_(self.linear_q_news.weight)
        nn.init.zeros_(self.linear_q_news.bias)

    def forward(self,
                batch_user_indices: torch.IntTensor,
                batch_his_title_indices: torch.IntTensor,
                batch_cand_title_indices: torch.IntTensor):
        """
        INPUTS:
            batch_user_indices: torch.IntTensor
                shape is B x 1

            batch_his_title_indices: torch.IntTensor
                shape is B x N x T, N is his_size

            batch_cand_title_indices: torch.IntTensor
                shape is B x N x T, N is candidate size
        """
        # Middle part
        batch_user_vec = self.user_embedding(batch_user_indices)  # B x user_emb_dim
        batch_query_word = self.linear_q_word(batch_user_vec)  # B x attention_hidden_dim
        batch_query_news = self.linear_q_news(batch_user_vec)  # B x attention_hidden_dim

        # Right-hand side part
        batch_clicked_news_repz = self.news_encoder(batch_his_title_indices, batch_query_word)  # B x N x filter_num
        batch_user_repz = self.news_attention_pool(batch_clicked_news_repz, batch_query_news)  # B x filter_num

        # Left-hand side part
        batch_cand_news_repz = self.news_encoder(batch_cand_title_indices, batch_query_word)  # B x N x filter_num

        # Click prediction
        batch_user_repz = batch_user_repz.unsqueeze(-1)  # B x filter_num x 1
        batch_logits = torch.bmm(batch_cand_news_repz, batch_user_repz)  # B x N x 1
        batch_logits = batch_logits.squeeze(dim=-1)  # B x N

        return batch_logits

    def score(self,
              batch_user_indices: torch.IntTensor,
              batch_his_title_indices: torch.IntTensor,
              batch_one_title_indices: torch.IntTensor
              ):
        """
        INPUTS:
            batch_user_indices: torch.IntTensor
                shape is B

            batch_his_title_indices: torch.IntTensor
                shape is B x N x T

            batch_one_title_indices: torch.IntTensor
                shape is B x 1 x T
        """
        batch_logits = self(batch_user_indices, batch_his_title_indices, batch_one_title_indices)  # B x 1
        batch_y_head = torch.sigmoid(batch_logits)  # B x 1
        return batch_y_head
