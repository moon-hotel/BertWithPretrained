import sys

sys.path.append('../')
from model.BertForClassification.BertForSequenceClassification import BertForSequenceClassification
from model.BasicBert.BertConfig import BertConfig
from utils.data_helpers import LoadClassificationDataset
from utils.log_helper import Logger
from transformers import BertTokenizer
import logging
import torch
import os
import time


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data')
        self.pretrained_model_dir = os.path.join(self.project_dir, "pretrained_model")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'toutiao_data_test.txt')
        self.val_file_path = os.path.join(self.dataset_dir, 'toutiao_data_test.txt')
        self.test_file_path = os.path.join(self.dataset_dir, 'toutiao_data_test.txt')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = '_!_'
        self.batch_size = 64
        self.max_sen_len = None
        self.num_labels = 15
        self.epochs = 10
        self.model_save_per_epoch = 2
        self.logger = Logger(log_file_name='test', log_level=logging.DEBUG, log_dir=self.logs_save_dir).get_log()

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        self.logger.info("\n\n\n\n\n########  <----------------------->")
        for key, value in self.__dict__.items():
            self.logger.info(f"########  {key} = {value}")


def train(config):
    classification_model = BertForSequenceClassification(config,
                                                         config.num_labels,
                                                         config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        classification_model.load_state_dict(loaded_paras)
        config.logger.info("## 成功载入已有模型，进行追加训练......")
    classification_model = classification_model.to(config.device)
    optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.0001)
    classification_model.train()
    bert_tokenize = BertTokenizer.from_pretrained(model_config.pretrained_model_dir).tokenize
    data_loader = LoadClassificationDataset(vocab_path=config.vocab_path,
                                            tokenizer=bert_tokenize,
                                            batch_size=config.batch_size,
                                            max_sen_len=config.max_sen_len,
                                            split_sep=config.split_sep,
                                            max_position_embeddings=config.max_position_embeddings,
                                            pad_index=config.pad_token_id)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = classification_model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            max_acc = max(acc, max_acc)
            if idx % 10 == 0:
                config.logger.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                              f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        config.logger.info(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_save_per_epoch == 0:
            acc = evaluate(test_iter, classification_model, config.device)
            config.logger.info(f"Accuracy on test {acc:.3f}")
            if acc > max_acc:
                torch.save(classification_model.state_dict(), model_save_path)


def inference(config):
    classification_model = BertForSequenceClassification(config,
                                                         config.num_labels,
                                                         config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        classification_model.load_state_dict(loaded_paras)
        config.logger.info("## 成功载入已有模型，进行追加训练......")
    classification_model = classification_model.to(config.device)
    data_loader = LoadClassificationDataset(vocab_path=config.vocab_path,
                                            tokenizer=BertTokenizer.from_pretrained(
                                                config.pretrained_model_dir).tokenize,
                                            batch_size=config.batch_size,
                                            max_sen_len=config.max_sen_len,
                                            split_sep=config.split_sep,
                                            max_position_embeddings=config.max_position_embeddings,
                                            pad_index=config.pad_token_id)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(config.train_file_path,
                                                                           config.val_file_path,
                                                                           config.test_file_path)
    acc = evaluate(test_iter, classification_model, device=config.device)
    config.logger.info(f"Acc on test:{acc:.3f}")


def evaluate(data_iter, model, device):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n


if __name__ == '__main__':
    model_config = ModelConfig()
    # train(model_config)
    # inference(model_config)
