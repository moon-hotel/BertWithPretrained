import sys

sys.path.append('../')
from utils import logger_init
import logging
from transformers import BertTokenizer
import os
from utils import LoadBertPretrainingDataset


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ========== wike2 数据集相关配置
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'WikiText')
        # self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        # self.train_file_path = os.path.join(self.dataset_dir, 'wiki.train.tokens')
        # self.val_file_path = os.path.join(self.dataset_dir, 'wiki.valid.tokens')
        # self.test_file_path = os.path.join(self.dataset_dir, 'wiki.test.tokens')
        # self.data_name = 'wiki2'
        # self.seps = "."

        # ========== songci 数据集相关配置
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SongCi')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.train_file_path = os.path.join(self.dataset_dir, 'songci.train.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'songci.valid.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'songci.test.txt')
        self.data_name = 'songci'
        self.seps = "。"

        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = None
        self.max_position_embeddings = 512
        self.pad_index = 0
        self.is_sample_shuffle = True
        self.random_state = 2021
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = logging.DEBUG

        logger_init(log_file_name=self.data_name, log_level=self.log_level,
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
                                             masked_token_unchanged_rate=model_config.masked_token_unchanged_rate,
                                             seps=model_config.seps)

    # train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(
    #     test_file_path=model_config.test_file_path,
    #     train_file_path=model_config.train_file_path,
    #     val_file_path=model_config.val_file_path)
    test_iter = data_loader.load_train_val_test_data(test_file_path=model_config.test_file_path,
                                                     only_test=True)
    for b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label in test_iter:
        print(b_token_ids.shape)  # [src_len,batch_size]
        print(b_segs.shape)  # [src_len,batch_size]
        print(b_mask.shape)  # [batch_size,src_len]
        print(b_mlm_label.shape)  # [src_len,batch_size]
        print(b_nsp_label.shape)  # [batch_size]
        token_ids = b_token_ids.transpose(0, 1)[4]
        mlm_label = b_mlm_label.transpose(0, 1)[4]
        token = " ".join([data_loader.vocab.itos[t] for t in token_ids])
        label = " ".join([data_loader.vocab.itos[t] for t in mlm_label])
        print(token)
        print(label)

        break

    sentences = ["十年生死两茫茫。不思量。自难忘。千里孤坟，无处话凄凉。",
                 "红酥手。黄藤酒。满园春色宫墙柳。"]
    token_ids, pred_idx, mask = data_loader.make_inference_samples(sentences,
                                                                   masked=False,
                                                                   language='zh',
                                                                   random_state=2022)
    print(token_ids.transpose(0, 1))
    print(pred_idx)
    print(mask)