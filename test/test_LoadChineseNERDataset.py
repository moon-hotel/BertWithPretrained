import sys

sys.path.append('../')
from utils.log_helper import logger_init
import logging
from transformers import BertTokenizer
from model import BertConfig
import os
from utils.data_helpers import LoadChineseNERDataset
import torch


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'ChineseNER')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'example_train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'example_dev.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'example_test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = ' '
        self.is_sample_shuffle = True
        self.batch_size = 5
        self.max_sen_len = None
        self.epochs = 10
        self.model_val_per_epoch = 2
        self.entities = {'O': 0, 'B-ORG': 1, 'B-LOC': 2, 'B-PER': 3, 'I-ORG': 4, 'I-LOC': 5, 'I-PER': 6}
        self.num_labels = len(self.entities)
        logger_init(log_file_name='ner', log_level=logging.DEBUG,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")
        self.__dict__["pad_token_id"] = -1


if __name__ == '__main__':
    model_config = ModelConfig()
    data_loader = LoadChineseNERDataset(
        entities=model_config.entities,
        num_labels=model_config.num_labels,
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(
            model_config.pretrained_model_dir).tokenize,
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle)
    test_iter = data_loader.load_train_val_test_data(train_file_path=model_config.train_file_path,
                                                     val_file_path=model_config.val_file_path,
                                                     test_file_path=model_config.test_file_path,
                                                     only_test=True)
    for sen, token_ids, labels in test_iter:
        print(sen)
        print(token_ids.shape)
        print(token_ids.transpose(0, 1))
        print(labels.shape)
        print(labels.transpose(0, 1))
        break
