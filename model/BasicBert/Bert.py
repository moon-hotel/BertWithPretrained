from .BertEmbedding import BertEmbeddings
from .MyTransformer import MyMultiheadAttention
import torch.nn as nn


# class BertModel(nn.Module):
#     def __init__(self, config):
#         self.config = config
#         self.embeddings = BertEmbeddings()


class BertSelfAttention(nn.Module):
    """
    实现多头注意力机制，对应的是GoogleResearch代码中的attention_layer方法
    https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L558
    """

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.multi_head_attention = MyMultiheadAttention(hidden_size=config.hidden_size,
                                                         num_heads=config.num_attention_heads,
                                                         dropout=config.attention_probs_dropout_prob)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """

        :param query: # [tgt_len, batch_size, hidden_size], tgt_len 表示目标序列的长度
        :param key:  #  [src_len, batch_size, hidden_size], src_len 表示源序列的长度
        :param value: # [src_len, batch_size, hidden_size], src_len 表示源序列的长度
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
        一般只在解码时使用，为了并行一次喂入所有解码部分的输入，所以要用mask来进行掩盖当前时刻之后的位置信息
        在Bert中，attention_mask指代的其实是key_padding_mask，因为Bert主要是基于Transformer Encoder部分构建的，
        所有没有Decoder部分，因此也就不需要用mask来进行掩盖当前时刻之后的位置信息
        :param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        :return:
        attn_output: [tgt_len, batch_size, hidden_size]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return self.multi_head_attention(query, key, value, attn_mask, key_padding_mask)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: 
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                hidden_states,  # [src_len, batch_size, hidden_size]
                attention_mask=None):  # [batch_size, src_len]
        self_outputs = self.self(hidden_states,
                                 hidden_states,
                                 hidden_states,
                                 attn_mask=None,
                                 key_padding_mask=attention_mask)
        # self_outputs[0] shape: [src_len, batch_size, hidden_size]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output
