"""
文件名: test/test_BertCLS.py
创建时间: 2023/2/28 8:59 上午
"""

import sys

sys.path.append("../")
from utils import logger_init
from model import BertConfig
from model import BertModel
from transformers import BertTokenizer
import os
import torch
import logging


#


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        logger_init(log_file_name='CLS', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value


if __name__ == '__main__':
    config = ModelConfig()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir)
    bert = BertModel.from_pretrained(config, config.pretrained_model_dir)
    sentences = ["各位朋友大家好。", "欢迎来到月来客栈。", "欢迎来到月来客栈。"]
    encode_input = bert_tokenize(sentences, return_tensors='pt', padding=True)
    input_ids = encode_input["input_ids"].transpose(0, 1)
    token_type_ids = encode_input["token_type_ids"].transpose(0, 1)
    attention_mask = encode_input["attention_mask"] == 0
    print(attention_mask)
    with torch.no_grad():
        bert.eval()
        pooled_output, _ = bert(input_ids, attention_mask)
        print("pooled_output:\n", pooled_output)


