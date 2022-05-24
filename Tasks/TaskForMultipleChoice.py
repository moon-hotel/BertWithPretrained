import sys

sys.path.append('../')
from model import BertConfig
from model import BertForMultipleChoice
from utils import LoadMultipleChoiceDataset
from utils import logger_init
from transformers import BertTokenizer
import logging
import torch
import os
import time
import numpy as np


class ModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'MultipleChoice')
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_uncased_english")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_file_path = os.path.join(self.dataset_dir, 'train.csv')
        self.val_file_path = os.path.join(self.dataset_dir, 'val.csv')
        self.test_file_path = os.path.join(self.dataset_dir, 'test.csv')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.is_sample_shuffle = True
        self.batch_size = 16
        self.max_sen_len = None
        self.num_labels = 4  # num_choice
        self.learning_rate = 2e-5
        self.epochs = 10
        self.model_val_per_epoch = 2
        logger_init(log_file_name='choice', log_level=logging.INFO,
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
            logging.info(f"### {key} = {value}")


def train(config):
    model = BertForMultipleChoice(config,
                                  config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(model_config.pretrained_model_dir).tokenize
    data_loader = LoadMultipleChoiceDataset(
        vocab_path=config.vocab_path,
        tokenizer=bert_tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle,
        num_choice=config.num_labels)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(config.train_file_path,
                                             config.val_file_path,
                                             config.test_file_path)
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (qa, seg, mask, label) in enumerate(train_iter):
            qa = qa.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            seg = seg.to(config.device)
            mask = mask.to(config.device)
            loss, logits = model(input_ids=qa,
                                 attention_mask=mask,
                                 token_type_ids=seg,
                                 position_ids=None,
                                 labels=label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
            if idx % 100 == 0:
                y_pred = logits.argmax(1).cpu()
                show_result(qa, y_pred, data_loader.vocab.itos, num_show=1)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch}, Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc, _ = evaluate(val_iter, model,
                              config.device, inference=False)
            logging.info(f"Accuracy on val {acc:.3f}")
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), model_save_path)


def inference(config):
    model = BertForMultipleChoice(config,
                                  config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadMultipleChoiceDataset(vocab_path=config.vocab_path,
                                            tokenizer=BertTokenizer.from_pretrained(
                                                config.pretrained_model_dir).tokenize,
                                            batch_size=config.batch_size,
                                            max_sen_len=config.max_sen_len,
                                            max_position_embeddings=config.max_position_embeddings,
                                            pad_index=config.pad_token_id,
                                            is_sample_shuffle=config.is_sample_shuffle)
    test_iter = data_loader.load_train_val_test_data(test_file_path=config.test_file_path,
                                                     only_test=True)
    y_pred = evaluate(test_iter, model, config.device, inference=True)
    logging.info(f"预测标签为：{y_pred.tolist()}")


def evaluate(data_iter, model, device, inference=False):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        y_pred = []
        for qa, seg, mask, y in data_iter:
            qa, seg, y, mask = qa.to(device), seg.to(device), y.to(device), mask.to(device)
            logits = model(qa, attention_mask=mask, token_type_ids=seg)
            y_pred.append(logits.argmax(1).cpu().numpy())
            if not inference:
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)
        model.train()
        if inference:
            return np.hstack(y_pred)
        return acc_sum / n, np.hstack(y_pred)


def show_result(qas, y_pred, itos=None, num_show=5):
    count = 0
    num_samples, num_choice, seq_len = qas.size()
    qas = qas.reshape(-1)
    strs = np.array([itos[t] for t in qas]).reshape(-1, seq_len)
    for i in range(num_samples):  # 遍历每个样本
        s_idx = i * num_choice
        e_idx = s_idx + num_choice
        sample = strs[s_idx:e_idx]
        if count == num_show:
            return
        count += 1
        for j, item in enumerate(sample):  # 每个样本的四个答案
            q, a, _ = " ".join(item[1:]).replace(" .", ".").replace(" ##", "").split('[SEP]')
            if y_pred[i] == j:
                a += " ## True"
            else:
                a += " ## False"
            logging.info(f"[{num_show}/{count}] ### {q + a}")
        logging.info("\n")


if __name__ == '__main__':
    model_config = ModelConfig()
    train(model_config)
    inference(model_config)
