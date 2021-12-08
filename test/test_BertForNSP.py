import sys

sys.path.append('../')
from model.DownstreamTasks import BertForNextSentencePrediction
from model.BasicBert.BertConfig import BertConfig
from utils.log_helper import logger_init
import logging
import os
import torch


class ModelConfig(object):
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        # 把原始bert中的配置参数也导入进来
        logger_init(log_file_name='bert_for_nsp', log_dir=self.logs_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def make_data():
    input_ids = torch.tensor([[1, 1, 1, 4, 5],  # [src_len, batch_size]
                              [6, 7, 8, 7, 2],
                              [5, 3, 4, 3, 4]]).transpose(0, 1)
    attention_mask = torch.tensor([[True, True, True, True, False],
                                   [True, True, True, False, False],
                                   [True, True, True, True, False]])
    # [batch_size, src_len] mask掉padding部分的内容

    token_type_ids = torch.tensor([[0, 0, 0, 1, 1],
                                   [0, 0, 1, 1, 0],
                                   [0, 0, 0, 1, 1]]).transpose(0, 1)
    # [src_len, batch_size] 如果输入模型的只有一个序列，那么这个参数也不用传值

    next_sentence_labels = torch.tensor([0, 1, 0])  # [batch_size,]
    return input_ids, attention_mask, token_type_ids, next_sentence_labels


if __name__ == '__main__':
    config = ModelConfig()
    input_ids, attention_mask, token_type_ids, next_sentence_labels = make_data()
    model = BertForNextSentencePrediction(config, config.pretrained_model_dir)
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids,
                   next_sentence_labels=None)
    print(output)
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids,
                   next_sentence_labels=next_sentence_labels)
    print(output)
