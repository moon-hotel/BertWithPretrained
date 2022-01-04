import os
import logging
import sys

sys.path.append('../')
from utils import logger_init
from model import BertConfig
from model import BertForPretrainingModel
from utils import LoadBertPretrainingDataset
from transformers import BertTokenizer
import torch
import time


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'WikiText')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'wiki.train.tokens')
        self.val_file_path = os.path.join(self.dataset_dir, 'wiki.valid.tokens')
        self.test_file_path = os.path.join(self.dataset_dir, 'wiki.test.tokens')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.is_sample_shuffle = True
        self.use_embedding_weight = False
        self.batch_size = 16
        self.max_sen_len = None
        self.pad_index = 0
        self.random_state = 2021
        self.data_name = 'wiki2'
        self.learning_rate = 3.5e-5
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.log_level = logging.INFO
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 2
        self.model_val_per_epoch = 1

        logger_init(log_file_name='wiki', log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    model = BertForPretrainingModel(config,
                                    config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             max_sen_len=config.max_sen_len,
                                             max_position_embeddings=config.max_position_embeddings,
                                             pad_index=config.pad_index,
                                             is_sample_shuffle=config.is_sample_shuffle,
                                             random_state=config.random_state,
                                             data_name=config.data_name,
                                             masked_rate=config.masked_rate,
                                             masked_token_rate=config.masked_token_rate,
                                             masked_token_unchanged_rate=config.masked_token_unchanged_rate)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                             train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            loss, mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                                 attention_mask=b_mask,
                                                 token_type_ids=b_segs,
                                                 masked_lm_labels=b_mlm_label,
                                                 next_sentence_labels=b_nsp_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            mlm_acc, _, _, nsp_acc, _, _ = accuracy(mlm_logits, nsp_logits, b_mlm_label,
                                                    b_nsp_label, data_loader.PAD_IDX)
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},"
                             f"nsp acc: {nsp_acc:.3f}")
            if idx % 100 == 0:
                pass
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        # if (epoch + 1) % config.model_val_per_epoch == 0:
        #     acc, _ = evaluate(val_iter, model,
        #                       config.device,
        #                       data_loader.PAD_IDX,
        #                       inference=False)
        #     logging.info(f" ### Accuracy on val: {round(acc, 4)}")
        #
        #     if acc > max_acc:
        #         max_acc = acc
        #     torch.save(model.state_dict(), config.model_save_path)


def accuracy(mlm_logits, nsp_logits, mlm_labels, nsp_label, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param nsp_logits:  [batch_size,2]
    :param nsp_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total

    nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    nsp_total = len(nsp_label)
    nsp_acc = float(nsp_correct) / nsp_total
    return mlm_acc, mlm_correct, mlm_total, nsp_acc, nsp_correct, nsp_total


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
