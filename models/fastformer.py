import torch
import torch.nn as nn

from models.base import BasicTorchModule
from models.utils import AdditiveAttentionPooling, init_embedding, EmbeddedPositionEncoding


class FastFormerNewsRecModel(BasicTorchModule):
    def __init__(self, hparams):
        super(FastFormerNewsRecModel, self).__init__(hparams)
        self.hparams = hparams
        self.news_encoder = NewsEncoder(hparams)
        self.user_encoder = UserEncoder(hparams)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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


class NewsEncoder(nn.Module):
    def __init__(self, hparams):
        super(NewsEncoder, self).__init__()
        self.d_model = hparams.d_model
        self.word_embedding = init_embedding(hparams.wordEmb_file)
        self.pos_encoder = EmbeddedPositionEncoding(hparams.word_emb_dim, hparams.title_size)
        self.layer_norm = nn.LayerNorm(hparams.d_model, eps=hparams.layer_norm_eps)
        self.dropout = nn.Dropout(hparams.dropout)
        self.seq_encoders = nn.ModuleList([FastFormerEncoder(hparams.d_model,
                                                             hparams.nhead,
                                                             hparams.intermediate_size,
                                                             hparams.layer_norm_eps,
                                                             hparams.dropout)
                                           for _ in range(hparams.num_encoder_layers)])
        self.attention_pooling = AdditiveAttentionPooling(hparams.d_model)

    def mask2score(self, mask: torch.BoolTensor):
        """
        input: bool tensor
        """
        mask = mask.float()
        score = (1 - mask) * -10000.0
        return score

    def get_input_repz(self, batch_ids):
        if batch_ids.dim() == 2:
            B, T = batch_ids.size()
            N = 1
        else:
            B, N, T = batch_ids.size()

        batch_ids = batch_ids.view(B*N, T)  # B*N x T
        mask = batch_ids.bool()  # B*N x T
        mask_scores = self.mask2score(mask)  # B*N x T
        embeds = self.word_embedding(batch_ids)   # B*N x T x word_emb_dim
        hidden_states = self.pos_encoder(embeds)  # B*N x T x word_emb_dim
        hidden_states = self.layer_norm(hidden_states)  # B*N x T x word_emb_dim
        hidden_states = self.dropout(hidden_states)  # B*N x T x word_emb_dim
        return hidden_states, mask_scores, B

    def forward(self, batch_news_title_indices):
        """
        Inputs:
            batch_news_title_indices: torch.FloatTensor
                shape is B x N x T or B x T
        """
        hidden_states, mask_scores, B = self.get_input_repz(batch_news_title_indices)
        extended_mask_scores = mask_scores.unsqueeze(1)  # B x 1 x T
        for seq_encoder in self.seq_encoders:
            hidden_states = seq_encoder(hidden_states, extended_mask_scores)  # B*N x T x d_model

        batch_news_repz = self.attention_pooling(hidden_states, mask_scores)  # B*N x d_model
        if batch_news_repz.size(0) != B:
            batch_news_repz = batch_news_repz.view(B, -1, self.d_model)
        return batch_news_repz


class UserEncoder(nn.Module):
    def __init__(self, hparams):
        super(UserEncoder, self).__init__()
        self.seq_encoders = nn.ModuleList([FastFormerEncoder(hparams.d_model,
                                                             hparams.nhead,
                                                             hparams.intermediate_size,
                                                             hparams.layer_norm_eps,
                                                             hparams.dropout)
                                           for _ in range(hparams.num_encoder_layers)])
        self.attention_pooling = AdditiveAttentionPooling(hparams.d_model)

    def forward(self, batch_candidate_news_repz):
        """
        Inputs:
            batch_candidate_news_repz: torch.FloatTensor
                the output of NewsEncoder and then reshape as B x N x d_model
        """
        B, N, D = batch_candidate_news_repz.size()
        mask_scores = torch.zeros([B, N])  # B x N
        extended_mask_scores = mask_scores.unsqueeze(1)  # B x 1 x N
        hidden_states = batch_candidate_news_repz  # B x N x d_model

        for seq_encoder in self.seq_encoders:
            hidden_states = seq_encoder(hidden_states, extended_mask_scores)  # B x N x d_model

        batch_user_repz = self.attention_pooling(hidden_states, mask_scores)  # B x d_model

        return batch_user_repz


class FastFormerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, intermediate_size: int, layer_norm_eps=1e-12, dropout=0.2):
        super(FastFormerEncoder, self).__init__()
        self.self_atten = FastSelfAttention(d_model, nhead)
        self.self_output_dense = nn.Linear(d_model, d_model)
        self.self_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.intermediate_dense = nn.Linear(d_model, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = nn.Linear(intermediate_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask_scores):
        attention_output = self.self_atten(hidden_states, mask_scores)  # B x T x d_model
        attention_output = self.self_output_dense(attention_output)
        attention_output = self.dropout(attention_output)  # B x T x d_model
        attention_output = self.self_layer_norm(attention_output)  # B x T x d_model

        intermediate_output = self.intermediate_dense(attention_output)  # B x T x intermediate_dim
        intermediate_output = self.intermediate_act_fn(intermediate_output)  # B x T x intermediate_dim

        output = self.output_dense(intermediate_output)  # B x T x d_model
        output = self.dropout(output)  # B x T x d_model
        # residual connect
        output = self.layer_norm(output + attention_output)  # B x T x d_model

        return output


class FastSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(FastSelfAttention, self).__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model must be a multiple of nhead")
        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = int(d_model // nhead)
        self.head_wq = nn.Parameter(torch.FloatTensor(nhead, self.head_dim))
        self.head_wk = nn.Parameter(torch.FloatTensor(nhead, self.head_dim))
        self.transform = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def get_additive_attention_score(self, head_vecs, head_weight, extended_mask_scores):
        """
        Inputs:
            head_vecs: torch.FloatTensor
                head query/key vectors which shape is B x nhead x T x head_dim

            head_weight: torch.FloatTensor
                the weight for calculating attention score for each head, the shape is nhead x head_dim

            extended_mask_scores: torch.FloatTensor
                score for each timestamp. -10000 for padding idx, otherwise is 0, the shape is B x 1 x T

        """
        head_weight = head_weight.unsqueeze(-1)  # nhead x head_dim x 1
        scores = head_vecs.matmul(head_weight) / self.head_dim ** 0.5  # B x nhead x T x 1
        scores = scores.squeeze(-1)  # B x nhead x T
        scores += extended_mask_scores  # B x nhead x T
        scores = self.softmax(scores)  # B x nhead x T

        return scores

    def forward(self, hidden_states, extended_mask_scores):
        """
        Inputs:
            hidden_states: torch.FloatTensor
                the input embedding or the output Fastformer encoder which shape is B x T x d_model

            extended_mask_scores: torch.FloatTensor
                score for each timestamp. -10000 for padding idx, otherwise is 0, the shape is B x 1 x T

        """
        B = hidden_states.size(0)
        # 1) Add all query vectors to a global query vector
        # Get all query vectors from hidden states
        q_vecs = self.WQ(hidden_states)  # B x T x d_model
        head_q_vecs = q_vecs.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)  # B x nhead x T x head_dim

        # alpha is the same notation in paper
        alpha = self.get_additive_attention_score(head_q_vecs, self.head_wq, extended_mask_scores)  # B x nhead x T
        alpha = alpha.unsqueeze(-2)  # B x nhead x 1 x T
        global_query = alpha.matmul(head_q_vecs)  # B x nhead x 1 x head_dim
        global_query = global_query.transpose(1, 2).view(B, 1, self.nhead * self.head_dim)  # B x 1 x d_model

        # 2) Add all key vectors to a global key vector
        # Get all key vectors from hidden states
        k_vecs = self.WK(hidden_states)  # B x T x d_model

        # Mixed the global query vector into all key vectors via element-wise dot product
        p_vecs = k_vecs * global_query  # B x T x d_model
        head_p_vecs = p_vecs.view(B, -1, self.nhead, self.head_dim).transpose(1, 2)  # B x nhead x T x head_dim

        # beta is the same notation in paper
        beta = self.get_additive_attention_score(head_p_vecs, self.head_wk, extended_mask_scores)  # B x nhead x T
        beta = beta.unsqueeze(-2)  # B x nhead x 1 x T
        global_key = beta.matmul(head_p_vecs)  # B x nhead x 1 x head_dim
        global_key = global_key.transpose(1, 2).view(B, 1, self.nhead * self.head_dim)  # B x 1 x d_model

        # 3) Get output vectors (notice that query == value in author's implementation)
        # Mixed the global key vector into all value vectors
        u_vecs = q_vecs * global_key  # B x T x d_model
        r_vecs = self.transform(u_vecs)  # B x T x d_model
        output_vecs = r_vecs + q_vecs

        return output_vecs










