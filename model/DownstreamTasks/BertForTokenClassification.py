from ..BasicBert.Bert import BertModel
import torch.nn as nn
import torch


class BertForTokenClassification(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForTokenClassification, self).__init__()
        self.num_labels = config.num_labels
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.config = config

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None):

        _, all_encoder_outputs = self.bert(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           position_ids=position_ids)  # [batch_size,hidden_size]
        sequence_output = all_encoder_outputs[-1]  # 取最后一层
        # sequence_output: [src_len, batch_size, hidden_size]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # logit: [src_len, batch_size, num_labels]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
