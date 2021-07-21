import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        #>>> pos_encoder = PositionalEmbedding(d_model)
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):  # [x_len, batch_size, d_model]
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = self.pe[:x.size(0), :]  # [src_len, 1, d_model]
        return x  # [src_len, 1, d_model]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.emb_size = emb_size

    """
        :param tokens: shape : [len, batch_size]
        :return: shape: [len, batch_size, emb_size]
        """

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(num_embeddings=2,
                         embedding_dim=embed_size)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, config):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=config.vocab_size,
                                    emb_size=config.hidden_size)
        # [src_len,batch_size,embed_size]
        self.position = PositionalEmbedding(d_model=config.hidden_size,
                                            max_len=config.max_position_embeddings)  # [src_len,1,embed_size]
        self.segment = SegmentEmbedding(embed_size=config.hidden_size)  # [src_len,batch_size,embed_size]
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        """
        :param input_ids: # [src_len,batch_size]
        :param token_type_ids: # [src_len,batch_size]
        :return:
        """
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.token(input_ids) + self.position(input_ids) + self.segment(token_type_ids)
        # [src_len,batch_size,embed_size] + [src_len,1,embed_size] + [src_len,batch_size,embed_size]
        return self.dropout(x)  # [src_len, batch_size, embed_size]
