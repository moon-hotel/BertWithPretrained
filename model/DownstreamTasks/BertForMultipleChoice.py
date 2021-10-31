from ..BasicBert.Bert import BertModel
import torch.nn as nn


class BertForMultipleChoice(nn.Module):
    """
    用于类似SWAG数据集的下游任务
    """
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMultipleChoice, self).__init__()
        self.num_choice = config.num_labels
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None):
        """

        :param input_ids: [batch_size, num_choice, src_len]
        :param attention_mask: [batch_size, num_choice, src_len]
        :param token_type_ids: [batch_size, num_choice, src_len]
        :param position_ids:
        :param labels:
        :return:
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)).transpose(0, 1)
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)).transpose(0, 1)
        flat_attention_mask = attention_mask.view(-1, token_type_ids.size(-1))

        pooled_output, _ = self.bert(
            input_ids=flat_input_ids,  # [src_len,batch_size*num_choice]
            attention_mask=flat_attention_mask,  # [batch_size*num_choice,src_len]
            token_type_ids=flat_token_type_ids,  # [src_len,batch_size*num_choice]
            position_ids=position_ids)
        pooled_output = self.dropout(pooled_output)  # [batch_size*num_choice, hidden_size]
        logits = self.classifier(pooled_output)  # [batch_size*num_choice, 1]
        shaped_logits = logits.view(-1, self.num_choice)  # [batch_size, num_choice]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shaped_logits, labels.view(-1))
            return loss, shaped_logits
        else:
            return shaped_logits
