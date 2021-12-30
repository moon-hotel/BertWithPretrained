import sys

sys.path.append('../')
from utils.log_helper import logger_init
import logging
from transformers import BertTokenizer
import os
from utils.create_pretraining_data import LoadBertPretrainingDataset


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'WikiText')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.train_file_path = os.path.join(self.dataset_dir, 'train.tokens')
        self.val_file_path = os.path.join(self.dataset_dir, 'val.tokens')
        self.test_file_path = os.path.join(self.dataset_dir, 'test.tokens')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = None
        self.max_position_embeddings = 512
        self.pad_index = 0
        self.is_sample_shuffle = True
        self.random_state = 2021
        self.data_name = 'wiki2'
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = logging.DEBUG

        logger_init(log_file_name='wiki', log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


if __name__ == '__main__':
    model_config = ModelConfig()
    data_loader = LoadBertPretrainingDataset(vocab_path=model_config.vocab_path,
                                             tokenizer=BertTokenizer.from_pretrained(
                                                 model_config.pretrained_model_dir).tokenize,
                                             batch_size=model_config.batch_size,
                                             max_sen_len=model_config.max_sen_len,
                                             max_position_embeddings=model_config.max_position_embeddings,
                                             pad_index=model_config.pad_index,
                                             is_sample_shuffle=model_config.is_sample_shuffle,
                                             random_state=model_config.random_state,
                                             data_name=model_config.data_name,
                                             masked_rate=model_config.masked_rate,
                                             masked_token_rate=model_config.masked_token_rate,
                                             masked_token_unchanged_rate=model_config.masked_token_unchanged_rate)

    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(
        test_file_path=model_config.test_file_path,
        train_file_path=model_config.train_file_path,
        val_file_path=model_config.val_file_path)
    for b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label in test_iter:
        print(b_token_ids.shape)  # [src_len,batch_size]
        print(b_segs.shape)  # [batch_size,src_len]
        print(b_mask.shape)  # [batch_size,src_len]
        print(b_mlm_label.shape)  # [src_len,batch_size]
        print(b_nsp_label.shape)  # [batch_size]
        break
