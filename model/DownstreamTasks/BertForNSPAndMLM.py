import logging
from ..BasicBert.Bert import BertModel
from ..BasicBert.Bert import get_activation
import torch.nn as nn
import torch


class BertForNextSentencePrediction(nn.Module):
    """
    仅为下一句预测模型
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForNextSentencePrediction, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size] 如果输入模型的只有一个序列，那么这个参数也不用传值
                position_ids=None,
                next_sentence_labels=None  # [batch_size,]
                ):
        pooled_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        # pooled_output: [batch_size, hidden_size]
        seq_relationship_score = self.classifier(pooled_output)
        # seq_relationship_score: [batch_size, 2]
        if next_sentence_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
            return loss
        else:
            return seq_relationship_score


class BertForLMTransformHead(nn.Module):
    """
    用于BertForMaskedLM中的一次变换。 因为在单独的MLM任务中
    和最后NSP与MLM的整体任务中均要用到，所以这里单独抽象为一个类便于复用

    ref: https://github.com/google-research/bert/blob/master/run_pretraining.py
        第248-262行
    """

    def __init__(self, config, bert_model_embedding_weights=None):
        """
        :param config:
        :param bert_model_embedding_weights:
        the output-weights are the same as the input embeddings, but there is
        an output-only bias for each token. 即TokenEmbedding层中的词表矩阵
        """
        super(BertForLMTransformHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        if bert_model_embedding_weights is not None:
            self.decoder.weight = nn.Parameter(bert_model_embedding_weights)
        # [hidden_size, vocab_size]
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, hidden_size] Bert最后一层的输出
        :return:
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.transform_act_fn(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.decoder(hidden_states)
        # hidden_states:  [src_len, batch_size, vocab_size]
        return hidden_states


class BertForMaskedLM(nn.Module):
    """
    仅为掩码语言预测模型
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMaskedLM, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        weights = None
        if config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            logging.info(f"## 使用token embedding中的权重矩阵作为输出层的权重！{weights.shape}")
        self.classifier = BertForLMTransformHead(config, weights)
        self.config = config

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None  # [src_len,batch_size]
                ):
        _, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        prediction_scores = self.classifier(sequence_output)
        # prediction_scores: [src_len, batch_size, vocab_size]
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.reshape(-1, self.config.vocab_size),
                                      masked_lm_labels.reshape(-1))
            return masked_lm_loss
        else:
            return prediction_scores  # [src_len, batch_size, vocab_size]


class BertForPretrainingModel(nn.Module):
    """
    BERT预训练模型，包括MLM和NSP两个任务
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForPretrainingModel, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 如果没有指定预训练模型路径，则随机初始化整个网络权重
            self.bert = BertModel(config)
        weights = None
        if config.use_embedding_weight:
            weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            logging.info(f"## 使用token embedding中的权重矩阵作为输出层的权重！{weights.shape}")
        self.mlm_prediction = BertForLMTransformHead(config, weights)
        self.nsp_prediction = nn.Linear(config.hidden_size, 2)
        self.config = config

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size]
                position_ids=None,
                masked_lm_labels=None,  # [src_len,batch_size]
                next_sentence_labels=None):  # [batch_size]
        pooled_output, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        mlm_prediction_score = self.mlm_prediction(sequence_output)
        # mlm_prediction_score: [src_len, batch_size, vocab_size]
        nsp_pred_score = self.nsp_prediciton(pooled_output)
        # nsp_pred_score： [batch_size, 2]
        if masked_lm_labels is not None and next_sentence_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            mlm_loss = loss_fct(mlm_prediction_score.reshape(-1, self.config.vocab_size),
                                masked_lm_labels.reshape(-1))
            nsp_loss = loss_fct(nsp_pred_score.reshape(-1, 2),
                                next_sentence_labels.reshape(-1))
            total_loss = mlm_loss + nsp_loss
            return total_loss
        else:
            return mlm_prediction_score, nsp_pred_score
        # [src_len, batch_size, vocab_size], [batch_size, 2]
