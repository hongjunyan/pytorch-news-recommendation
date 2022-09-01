import math
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, head_num: int, head_dim: int):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.head_dim = head_dim
        self.output_dim = head_num * head_dim
        self.WQ = nn.Linear(input_dim, self.output_dim, bias=False)
        self.WK = nn.Linear(input_dim, self.output_dim, bias=False)
        self.WV = nn.Linear(input_dim, self.output_dim, bias=False)

        nn.init.xavier_normal_(self.WQ.weight)
        nn.init.xavier_normal_(self.WK.weight)
        nn.init.xavier_normal_(self.WV.weight)

    def forward(self, query, key, value):
        """
        INPUTS:
            query/key/value: torch.FloatTensor
                - shape is B x N x T x word_emb_dim for news_encoder
                - shape is B x N x attention_hidden_dim for user_encoder
        RETURN:
            attention vectors: torch.FloatTensor
                shape is B x N x T x word_emb_dim
        """
        if query.dim() == 4:
            # For news_encoder
            B, N = query.size(0), query.size(1)
            # After linear transform, the shape of query/key/value is B x N x H x T x head_dim
            query, key, value = [linear(x).view(B, N, -1, self.head_num, self.head_dim).transpose(-2, -3)
                                 for linear, x in zip((self.WQ, self.WK, self.WV), (query, key, value))]

            batch_repzs = additive_attention(query, key, value)  # B x N x H x T x head_dim
            batch_repzs = batch_repzs.transpose(-2, -3).contiguous()  # B x N x T x H x head_dim
            batch_repzs = batch_repzs.view(B, N, -1, self.head_num * self.head_dim)  # B x N x T x output_dim
        else:
            # For user_encoder
            B = query.size(0)
            # After linear transform, the shape of query/key/value is B x H x N x head_dim
            query, key, value = [linear(x).view(B, -1, self.head_num, self.head_dim).transpose(1, 2)
                                 for linear, x in zip((self.WQ, self.WK, self.WV), (query, key, value))]

            batch_repzs = additive_attention(query, key, value)  # B x H x N x head_dim
            batch_repzs = batch_repzs.transpose(1, 2).contiguous()  # B x N x H x head_dim
            batch_repzs = batch_repzs.view(B, -1, self.head_num * self.head_dim)  # B x N x output_dim

        return batch_repzs


class AttLayer(nn.Module):
    def __init__(self, input_dim: int, attention_hidden_dim: int):
        super(AttLayer, self).__init__()
        self.input_dim = input_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.linear = nn.Linear(input_dim, attention_hidden_dim)
        self.query = torch.nn.Parameter(torch.FloatTensor(attention_hidden_dim, 1))  # default all elements are zeros
        nn.init.xavier_normal_(self.query)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, batch_repzs: torch.FloatTensor):
        """
        INPUTS:
            batch_repzs: torch.FloatTensor,
                - shape is B x N x T x (head_num * head_dim) for news_encoder
                - shape is B x N x (head_num * head_dim) for user_encoder
        OUTPUT:
            output: torch.FloatTensor,
                - shape is B x N x (head_num * head_dim) for news_encoder
                - shape is B x (head_num * head_dim) for user_enocder
        """
        batch_repzs_ = self.linear(batch_repzs)  # B x N x T x attention_hidden_dim
        batch_repzs_ = torch.tanh(batch_repzs_)  # B x N x T x attention_hidden_dim
        batch_scores = torch.matmul(batch_repzs_, self.query)  # B x N x T x 1
        batch_scores = torch.softmax(batch_scores, dim=-2)  # softmax along T axis, shape is B x N x T x 1

        # weighted sum of input tensors
        batch_output = torch.mul(batch_repzs, batch_scores)  # B x N x T x (head_num * head_dim)
        batch_output = batch_output.sum(dim=-2)  # B x N x (head_num * head_dim)
        return batch_output


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
        batch_value_repz = self.dropout(batch_value_repz)  # B x N x T x filter_num

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
    # the first dim is 0 for padding
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


def additive_attention(query, key, value):
    """
    INPUTS:
        query/key/value: torch.FloatTensor
            shape is B x N x H x T x head_dim, where attention_hidden_dim is H * head_dim
    RETURN:
        attention vectors: torch.FloatTensor
            shape is B x N x H x T x head_dim
    """
    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)  # B x N x H x T x T
    scores = torch.softmax(scores, dim=-1)  # B x N x H x T x T
    batch_repzs = torch.matmul(scores, value)  # B x N x H x T x head_dim
    return batch_repzs
