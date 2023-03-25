import torch
from torch.nn.init import normal_
from .BertEmbedding import BertEmbeddings
from .MyTransformer import MyMultiheadAttention
import torch.nn as nn
import os
import logging
from copy import deepcopy


def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % act)


class BertSelfAttention(nn.Module):
    """
    实现多头注意力机制，对应的是GoogleResearch代码中的attention_layer方法
    https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L558
    """

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            MultiHeadAttention = nn.MultiheadAttention
        else:
            MultiHeadAttention = MyMultiheadAttention
        self.multi_head_attention = MultiHeadAttention(embed_dim=config.hidden_size,
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
        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        :param hidden_states: [src_len, batch_size, hidden_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        # hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        self_outputs = self.self(hidden_states,
                                 hidden_states,
                                 hidden_states,
                                 attn_mask=None,
                                 key_padding_mask=attention_mask)
        # self_outputs[0] shape: [src_len, batch_size, hidden_size]
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, intermediate_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, intermediate_size]
        if self.intermediate_act_fn is None:
            hidden_states = hidden_states
        else:
            hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """

        :param hidden_states: [src_len, batch_size, intermediate_size]
        :param input_tensor: [src_len, batch_size, hidden_size]
        :return: [src_len, batch_size, hidden_size]
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_attention = BertAttention(config)
        self.bert_intermediate = BertIntermediate(config)
        self.bert_output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len] mask掉padding部分的内容
        :return: [src_len, batch_size, hidden_size]
        """
        attention_output = self.bert_attention(hidden_states, attention_mask)
        # [src_len, batch_size, hidden_size]
        intermediate_output = self.bert_intermediate(attention_output)
        # [src_len, batch_size, intermediate_size]
        layer_output = self.bert_output(intermediate_output, attention_output)
        # [src_len, batch_size, hidden_size]
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None):
        """

        :param hidden_states: [src_len, batch_size, hidden_size]
        :param attention_mask: [batch_size, src_len]
        :return:
        """
        all_encoder_layers = []
        layer_output = hidden_states
        for i, layer_module in enumerate(self.bert_layers):
            layer_output = layer_module(layer_output,
                                        attention_mask)
            #  [src_len, batch_size, hidden_size]
            all_encoder_layers.append(layer_output)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        """

        :param hidden_states:  [src_len, batch_size, hidden_size]
        :return: [batch_size, hidden_size]
        """
        if 'pooler_type' not in self.config.__dict__:
            raise ValueError("pooler_type must be in ['first_token_transform', 'all_token_average']"
                             "请在配置文件config.json中添加一个pooler_type参数")
        if self.config.pooler_type == "first_token_transform":
            token_tensor = hidden_states[0, :].reshape(-1, self.config.hidden_size)
        elif self.config.pooler_type == "all_token_average":
            token_tensor = torch.mean(hidden_states, dim=0)
        pooled_output = self.dense(token_tensor)  # [batch_size, hidden_size]
        pooled_output = self.activation(pooled_output)
        return pooled_output  # [batch_size, hidden_size]


def format_paras_for_torch(loaded_paras_names, loaded_paras):
    """
    该函数的作用是将预训练参数格式化成符合torch(1.5.0)框架中MultiHeadAttention的参数形式
    :param loaded_paras_names:
    :param loaded_paras:
    :return:
    """
    qkv_weight_names = ['query.weight', 'key.weight', 'value.weight']
    qkv_bias_names = ['query.bias', 'key.bias', 'value.bias']
    qkv_weight, qkv_bias = [], []
    torch_paras = []
    for i in range(len(loaded_paras_names)):
        para_name_in_pretrained = loaded_paras_names[i]
        para_name = ".".join(para_name_in_pretrained.split('.')[-2:])
        if para_name in qkv_weight_names:
            qkv_weight.append(loaded_paras[para_name_in_pretrained])
        elif para_name in qkv_bias_names:
            qkv_bias.append(loaded_paras[para_name_in_pretrained])
        else:
            torch_paras.append(loaded_paras[para_name_in_pretrained])
        if len(qkv_weight) == 3:
            torch_paras.append(torch.cat(qkv_weight, dim=0))
            qkv_weight = []
        if len(qkv_bias) == 3:
            torch_paras.append(torch.cat(qkv_bias, dim=0))
            qkv_bias = []
    return torch_paras


def replace_512_position(init_embedding, loaded_embedding):
    """
    本函数的作用是当max_positional_embedding > 512时，用预训练模型中的512个向量来
    替换随机初始化的positional embedding中的前512个向量
    :param init_embedding:  初始化的positional embedding矩阵，大于512行
    :param loaded_embedding: 预训练模型中的positional embedding矩阵，等于512行
    :return: 前512行被替换后的初始化的positional embedding矩阵
    """
    logging.info(f"模型参数max_positional_embedding > 512，采用替换处理！")
    init_embedding[:512, :] = loaded_embedding[:512, :]
    return init_embedding


class BertModel(nn.Module):
    """

    """

    def __init__(self, config):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)
        self.bert_encoder = BertEncoder(config)
        self.bert_pooler = BertPooler(config)
        self.config = config
        self._reset_parameters()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None):
        """
        ***** 一定要注意，attention_mask中，被mask的Token用1(True)表示，没有mask的用0(false)表示
        这一点一定一定要注意
        :param input_ids:  [src_len, batch_size]
        :param attention_mask: [batch_size, src_len] mask掉padding部分的内容
        :param token_type_ids: [src_len, batch_size]  # 如果输入模型的只有一个序列，那么这个参数也不用传值
        :param position_ids: [1,src_len] # 在实际建模时这个参数其实可以不用传值
        :return:
        """
        embedding_output = self.bert_embeddings(input_ids=input_ids,
                                                position_ids=position_ids,
                                                token_type_ids=token_type_ids)
        # embedding_output: [src_len, batch_size, hidden_size]
        all_encoder_outputs = self.bert_encoder(embedding_output,
                                                attention_mask=attention_mask)
        # all_encoder_outputs 为一个包含有num_hidden_layers个层的输出
        sequence_output = all_encoder_outputs[-1]  # 取最后一层
        # sequence_output: [src_len, batch_size, hidden_size]
        pooled_output = self.bert_pooler(sequence_output)
        # 默认是最后一层的first token 即[cls]位置经dense + tanh 后的结果
        # pooled_output: [batch_size, hidden_size]
        return pooled_output, all_encoder_outputs

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=self.config.initializer_range)

    @classmethod
    def from_pretrained(cls, config, pretrained_model_dir=None):
        model = cls(config)  # 初始化模型，cls为未实例化的对象，即一个未实例化的BertModel对象
        pretrained_model_path = os.path.join(pretrained_model_dir, "pytorch_model.bin")
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"<路径：{pretrained_model_path} 中的模型不存在，请仔细检查！>\n"
                             f"中文模型下载地址：https://huggingface.co/bert-base-chinese/tree/main\n"
                             f"英文模型下载地址：https://huggingface.co/bert-base-uncased/tree/main\n")
        loaded_paras = torch.load(pretrained_model_path)
        state_dict = deepcopy(model.state_dict())
        loaded_paras_names = list(loaded_paras.keys())[:-8]
        model_paras_names = list(state_dict.keys())[1:]
        if 'use_torch_multi_head' in config.__dict__ and config.use_torch_multi_head:
            torch_paras = format_paras_for_torch(loaded_paras_names, loaded_paras)
            for i in range(len(model_paras_names)):
                logging.debug(f"## 成功赋值参数:{model_paras_names[i]},形状为: {torch_paras[i].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    # 这部分代码用来消除预训练模型只能输入小于512个字符的限制
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]])
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                state_dict[model_paras_names[i]] = torch_paras[i]
            logging.info(f"## 注意，正在使用torch框架中的MultiHeadAttention实现")
        else:
            for i in range(len(loaded_paras_names)):
                logging.debug(f"## 成功将参数:{loaded_paras_names[i]}赋值给{model_paras_names[i]},"
                              f"参数形状为:{state_dict[model_paras_names[i]].size()}")
                if "position_embeddings" in model_paras_names[i]:
                    # 这部分代码用来消除预训练模型只能输入小于512个字符的限制
                    if config.max_position_embeddings > 512:
                        new_embedding = replace_512_position(state_dict[model_paras_names[i]],
                                                             loaded_paras[loaded_paras_names[i]])
                        state_dict[model_paras_names[i]] = new_embedding
                        continue
                state_dict[model_paras_names[i]] = loaded_paras[loaded_paras_names[i]]
            logging.info(f"## 注意，正在使用本地MyTransformer中的MyMultiHeadAttention实现，"
                         f"如需使用torch框架中的MultiHeadAttention模块可通过config.__dict__['use_torch_multi_head'] = True实现")
        model.load_state_dict(state_dict)
        return model
