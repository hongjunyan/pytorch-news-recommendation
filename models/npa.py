import torch
import torch.nn as nn
from models.base import BasicTorchModule
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

    def forward(self, batch_input: tuple):
        """
        INPUTS:
            batch_input: Tuple,
                batch_user_indices: torch.IntTensor
                    shape is B x 1

                batch_his_title_indices: torch.IntTensor
                    shape is B x N x T, N is his_size

                batch_cand_title_indices: torch.IntTensor
                    shape is B x N x T, N is candidate size
        """
        batch_user_indices, batch_his_title_indices, batch_cand_title_indices = batch_input
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

    def score(self, batch_input: tuple):
        """
        INPUTS:
            batch_input, Tuple,
                batch_user_indices: torch.IntTensor
                    shape is B

                batch_his_title_indices: torch.IntTensor
                    shape is B x N x T

                batch_one_title_indices: torch.IntTensor
                    shape is B x 1 x T
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
