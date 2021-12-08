from ..BasicBert.Bert import BertModel
import torch.nn as nn


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
