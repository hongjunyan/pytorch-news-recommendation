import math
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np


class PersonalizedAttentivePooling(nn.Module):
    def __init__(self, value_dim: int, attention_hidden_dim: int):
        """
        INPUTS:
            value_num: num of value
                - number of words for NewsEncoder
                - number of news for UserEncoder

            value_dim: dimension of value
                - word_emb_dim for NewsEncoder
                - attention_hidden_dim for UserEncoder

            attention_hidden_dim: dimension of attention,
                we convert all vectors to this dim for applying attention mechanism
        """
        super(PersonalizedAttentivePooling, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(value_dim, attention_hidden_dim)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, batch_value_repz: torch.FloatTensor, batch_queries: torch.FloatTensor) -> torch.FloatTensor:
        """
        INPUTS:
            batch_values_repz: torch.FloatTensor
                - shape is B x N x T x filter_num for NewsEncoder
                - shape is B x N x filter_num for UserEncoder

            batch_queries: torch.FloatTensor
                shape is B x attention_hidden_dim

        RETURNS:
            batch_output: torch.FloatTensor
                - shape is B x N x filter_num for NewsEncoder
                - shape is B x filter_num for UserEncoder

        """
        # dropout to prevent overfitting
        batch_value_repz = self.dropout(batch_value_repz)

        # convert value_repz to atten_vec
        batch_atten_vecs = self.linear(batch_value_repz)  # B x N x T x attention_hidden_dim
        batch_atten_vecs = torch.tanh(batch_atten_vecs)  # B x N x T x attention_hidden_dim

        # flatten along B*N
        atten_shape = batch_atten_vecs.shape
        B, N, attention_hidden_dim = atten_shape[0], atten_shape[1], atten_shape[-1]
        batch_atten_vecs = batch_atten_vecs.view(B*N, -1, attention_hidden_dim)  # (B*N) x T x attention_hidden_dim
        batch_queries = torch.repeat_interleave(batch_queries, N, dim=0)  # (B*N) x attention_hidden_dim
        batch_queries = batch_queries.unsqueeze(-1)  # (B*N) x attention_hidden_dim x 1

        # calculate attention scores
        batch_scores = torch.bmm(batch_atten_vecs, batch_queries)  # (B*N) x T x 1

        # reconstruct B, N
        batch_scores = batch_scores.view(B, N, -1).squeeze(dim=-1)  # B x N x T
        batch_scores = torch.softmax(batch_scores, dim=-1)  # B x N x T
        batch_scores = batch_scores.unsqueeze(-1)  # B x N x T x 1

        # get attention vectors
        batch_output = torch.mul(batch_value_repz, batch_scores)  # B x N x T x filter_num
        batch_output = batch_output.sum(dim=-2)  # B x N x filter_num

        return batch_output


def init_embedding(npy_file: Path) -> torch.nn.Embedding:
    pretrain_word_emb_npy = np.load(str(npy_file))
    weight = torch.FloatTensor(pretrain_word_emb_npy)
    # set freeze is False, because we want embedding trained with the whole network
    return nn.Embedding.from_pretrained(weight, freeze=False)


def init_conv1d(in_channels: int, out_channels: int, kernel_size=3):
    # always used "SAME" padding, i.e., the dimension of input and output are the same.
    padding = math.floor(kernel_size / 2)
    conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    nn.init.xavier_normal_(conv1d.weight)
    nn.init.zeros_(conv1d.bias)
    return conv1d
