from ..BasicBert.Bert import BertModel
import torch.nn as nn


class BertForQuestionAnswering(nn.Module):
    """
    用于建模类似SQuAD这样的问答数据集
    """

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForQuestionAnswering, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:
            self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                start_positions=None,
                end_positions=None):
        """
        :param input_ids: [src_len,batch_size]
        :param attention_mask: [batch_size,src_len]
        :param token_type_ids: [src_len,batch_size]
        :param position_ids:
        :param start_positions: [batch_size]
        :param end_positions:  [batch_size]
        :return:
        """
        _, all_encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = all_encoder_outputs[-1]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        logits = self.qa_outputs(sequence_output)  # [src_len, batch_size,2]
        start_logits, end_logits = logits.split(1, dim=-1)
        # [src_len,batch_size,1]  [src_len,batch_size,1]
        start_logits = start_logits.squeeze(-1).transpose(0, 1)  # [batch_size,src_len]
        end_logits = end_logits.squeeze(-1).transpose(0, 1)  # [batch_size,src_len]
        if start_positions is not None and end_positions is not None:
            # 由于部分情况下start/end 位置会超过输入的长度
            # （例如输入序列的可能大于512，并且正确的开始或者结束符就在512之后）
            # 那么此时就要进行特殊处理
            ignored_index = start_logits.size(1)  # 取输入序列的长度
            start_positions.clamp_(0, ignored_index)
            # 如果正确起始位置start_positions中，存在输入样本的开始位置大于输入长度，
            # 那么直接取输入序列的长度作为开始位置
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            # 这里指定ignored_index其实就是为了忽略掉超过输入序列长度的（起始结束）位置
            # 在预测时所带来的损失，因为这些位置并不能算是模型预测错误的（只能看做是没有预测），
            # 同时如果不加ignore_index的话，那么可能会影响模型在正常情况下的语义理解能力
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            return (start_loss + end_loss) / 2, start_logits, end_logits
        else:
            return start_logits, end_logits  # [batch_size,src_len]
