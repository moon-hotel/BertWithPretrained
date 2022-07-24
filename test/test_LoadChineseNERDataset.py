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
        self.ignore_idx = -100
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


if __name__ == '__main__':
    model_config = ModelConfig()
    data_loader = LoadChineseNERDataset(
        entities=model_config.entities,
        num_labels=model_config.num_labels,
        ignore_idx=model_config.ignore_idx,
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
        # ['我们变而以书会友，以书结缘，把欧美、港台流行的食品类图谱、画册、工具书汇集一堂。',
        # '为了跟踪国际最新食品工艺、流行趋势，大量搜集海外专业书刊资料是提高技艺的捷径。', ...']

        print("Input Token:\n", token_ids.shape)
        # tensor([[ 101, 2769,  812, 1359, 5445,  809,  741,  833, 1351, 8024,  809,  741,
        #          5310, 5357, 8024, 2828, 3616, 5401,  510, 3949, 1378, 3837, 6121, 4638,
        #          7608, 1501, 5102, 1745, 6480,  510, 4514, 1085,  510, 2339, 1072,  741,
        #          3726, 7415,  671, 1828,  511,  102,    0,    0,    0],
        #         [ 101,  711,  749, 6656, 6679, 1744, 7354, 3297, 3173, 7608, 1501, 2339,
        #          5686,  510, 3837, 6121, 6633, 1232, 8024, 1920, 7030, 3017, 7415, 3862,
        #          1912,  683,  689,  741, 1149, 6598, 3160, 3221, 2990, 7770, 2825, 5686,
        #          4638, 2949, 2520,  511,  102,    0,    0,    0,    0],....]
        print(token_ids.transpose(0, 1))
        print("Attention Mask（Padding Mask): \n", (token_ids == model_config.pad_token_id).transpose(0, 1))
        # tensor([[False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False,  True,  True,  True],
        #         [False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False, False, False, False, False, False, False, False, False, False,
        #          False,  True,  True,  True,  True],....]
        print("Labels:\n", labels.shape)  # [src_len,batch_size]
        print(labels.transpose(0, 1))
        # tensor([[-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        #             0,    0,    0,    0,    2,    2,    0,    2,    2,    0,    0,    0,
        #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        #             0,    0,    0,    0,    0, -100, -100, -100, -100],
        #         [-100,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        #             0,    0,    0,    0, -100, -100, -100, -100, -100],....]
        break

    sentences = ['智光拿出石壁拓文为乔峰详述事情始末，乔峰方知自己原本姓萧，乃契丹后族。',
                 '当乔峰问及带头大哥时，却发现智光大师已圆寂。',
                 '乔峰、阿朱相约找最后知情人康敏问完此事后，就到塞外骑马牧羊，再不回来。']
    batch_sentence, batch_token_ids, _ = data_loader.make_inference_samples(sentences)
    print("============")
    print(batch_sentence)
    print(batch_token_ids)
