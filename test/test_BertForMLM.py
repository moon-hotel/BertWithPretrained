import sys

sys.path.append('../')
from model import BertForMaskedLM
from model import BertConfig
from utils import logger_init
import logging
import os
import torch


class ModelConfig(object):
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.use_embedding_weight = True  # 是否使用Token embedding中的权重作为预测时输出层的权重
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
    import numpy as np
    ids = np.random.random_integers(0, 300, 512 * 3).reshape(3, 512)
    input_ids = torch.tensor(ids).transpose(0, 1)
    labels = np.random.random_integers(0, 1, 512 * 3).reshape(3, 512)
    masked_lm_labels = torch.tensor(labels, dtype=torch.long).transpose(0, 1)  # [src_len,batch_size]
    return input_ids, masked_lm_labels


if __name__ == '__main__':
    config = ModelConfig()
    input_ids, masked_lm_labels = make_data()
    model = BertForMaskedLM(config, config.pretrained_model_dir)
    output = model(input_ids=input_ids,
                   masked_lm_labels=None)
    print(output)
    output = model(input_ids=input_ids,
                   masked_lm_labels=masked_lm_labels)
    print(output)
