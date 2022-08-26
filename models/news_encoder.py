import argparse
import torch
import torch.nn as nn
from models.utils import PersonalizedAttentivePooling, init_embedding, init_conv1d


class NewsEncoder(nn.Module):
    def __init__(self, hparams):
        super(NewsEncoder, self).__init__()
        self.hparams = hparams

        self.dropout = nn.Dropout(p=hparams.dropout)
        self.word_embedding = init_embedding(hparams.wordEmb_file)
        self.conv1d = init_conv1d(hparams.word_emb_dim, hparams.filter_num, hparams.window_size)
        self.cnn_activation = self._get_cnn_activation()
        self.word_attention_pool = PersonalizedAttentivePooling(
            hparams.filter_num,
            hparams.attention_hidden_dim
        )

    def _get_cnn_activation(self):
        if self.hparams.cnn_activation == "relu":
            return nn.ReLU()

    def cnn_text_encoder(self, batch_news_word_embedding: torch.FloatTensor):
        """
        INPUTS:
            batch_news_word_embedding: torch.FloatTensor
                shape is B x N x T x word_emb_dim

        RETURNS:
            batch_news_word_repz, torch.FloatTensor
                shape is B x N x T x filter_num
        """
        B, N, T, word_emb_dim = batch_news_word_embedding.shape
        # dropout word_embedding to prevent overfitting
        batch_news_word_embedding = self.dropout(batch_news_word_embedding)

        # flatten batch
        batch_news_word_embedding = batch_news_word_embedding.view(-1, T, word_emb_dim)  # (B*N) x T x word_emb_dim

        # transpose T and word_emb_dim
        batch_news_word_embedding = batch_news_word_embedding.transpose(-2, -1)  # (B*N) x word_emb_dim x T

        # title encoding by using CNN
        batch_news_word_repz = self.conv1d(batch_news_word_embedding)  # (B*N) x num_filter x T
        batch_news_word_repz = self.cnn_activation(batch_news_word_repz)

        # reconstruct batch
        batch_news_word_repz = batch_news_word_repz.view(B, N, -1, T)  # B x N x num_filter x T

        # transpose word_emb_dim and T
        batch_news_word_repz = batch_news_word_repz.transpose(-2, -1)  # B x N x T x num_filter

        # dropout num_filter to prevent overfitting
        batch_news_word_repz = self.dropout(batch_news_word_repz)

        return batch_news_word_repz

    def forward(self, batch_news_indices: torch.IntTensor, batch_query: torch.FloatTensor):
        """
        INPUTS:
            batch_news: torch.IntTensor,
                shape is B x N x T, where
                    B is Batch
                    N is Number_of_candidates or Number of user history
                    T is the length of title, usually called timestamp

            batch_query: torch.FloatTensor
                shape is B x attention_hidden_dim

        """
        batch_news_word_embedding = self.word_embedding(batch_news_indices)  # B x N x T x word_emb_dim
        batch_news_word_repz = self.cnn_text_encoder(batch_news_word_embedding)  # B x N x T x filter_num
        batch_news_repz = self.word_attention_pool(batch_news_word_repz, batch_query)  # B x N x filter_num

        return batch_news_repz
