import sys
from copy import deepcopy

sys.path.append('../')
from transformers import BertTokenizer
from model import BertConfig
from model import BertForTokenClassification
from utils import LoadChineseNERDataset
from utils import logger_init
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report
import logging
import os
import torch
import time


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
        self.model_save_name = "ner_model.pt"
        self.writer = SummaryWriter("runs")
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.split_sep = ' '
        self.is_sample_shuffle = True
        self.batch_size = 12
        self.max_sen_len = None
        self.epochs = 10
        self.learning_rate = 1e-5
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


def accuracy(logits, y_true, ignore_idx=-100):
    """
    :param logits:  [src_len,batch_size,num_labels]
    :param y_true:  [src_len,batch_size]
    :param ignore_idx: 默认情况为-100
    :return:
    e.g.
    y_true = torch.tensor([[-100, 0, 0, 1, -100],
                       [-100, 2, 0, -100, -100]]).transpose(0, 1)
    logits = torch.tensor([[[0.5, 0.1, 0.2], [0.5, 0.4, 0.1], [0.7, 0.2, 0.3], [0.5, 0.7, 0.2], [0.1, 0.2, 0.5]],
                           [[0.3, 0.2, 0.5], [0.7, 0.2, 0.4], [0.8, 0.1, 0.3], [0.9, 0.2, 0.1], [0.1, 0.5, 0.2]]])
    logits = logits.transpose(0, 1)
    print(accuracy(logits, y_true, -100)) # (0.8, 4, 5)
    """
    y_pred = logits.transpose(0, 1).argmax(axis=2).reshape(-1).tolist()
    # 将 [src_len,batch_size,num_labels] 转成 [batch_size, src_len,num_labels]
    y_true = y_true.transpose(0, 1).reshape(-1).tolist()
    real_pred, real_true = [], []
    for item in zip(y_pred, y_true):
        if item[1] != ignore_idx:
            real_pred.append(item[0])
            real_true.append(item[1])
    return accuracy_score(real_true, real_pred), real_true, real_pred


def train(config):
    model = BertForTokenClassification(config,
                                       config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir,
                                   config.model_save_name)
    global_steps = 0
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        global_steps = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")

    data_loader = LoadChineseNERDataset(
        entities=config.entities,
        num_labels=config.num_labels,
        ignore_idx=config.ignore_idx,
        vocab_path=config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(
            config.pretrained_model_dir).tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        split_sep=config.split_sep,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = \
        data_loader.load_train_val_test_data(train_file_path=config.train_file_path,
                                             val_file_path=config.val_file_path,
                                             test_file_path=config.test_file_path,
                                             only_test=False)
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    max_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sen, token_ids, labels) in enumerate(train_iter):
            token_ids = token_ids.to(config.device)
            labels = labels.to(config.device)
            padding_mask = (token_ids == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(input_ids=token_ids,  # [src_len, batch_size]
                                 attention_mask=padding_mask,  # [batch_size,src_len]
                                 token_type_ids=None,
                                 position_ids=None,
                                 labels=labels)  # [src_len, batch_size]
            # logit: [src_len, batch_size, num_labels]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            global_steps += 1
            acc, _, _ = accuracy(logits, labels, config.ignore_idx)
            if idx % 20 == 0:
                logging.info(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train acc: {round(acc, 5)}")
                config.writer.add_scalar('Training/Loss', loss.item(), global_steps)
                config.writer.add_scalar('Training/Acc', acc, global_steps)
            if idx % 100 == 0:
                show_result(sen[:10], logits[:, :10], token_ids[:, :10], config.entities)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}],"
                     f" Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_val_per_epoch == 0:
            acc = evaluate(config, val_iter, model, data_loader)
            logging.info(f"Accuracy on val {acc:.3f}")
            config.writer.add_scalar('Testing/Acc', acc, global_steps)
            if acc > max_acc:
                max_acc = acc
                state_dict = deepcopy(model.state_dict())
                torch.save({'last_epoch': global_steps,
                            'model_state_dict': state_dict},
                           model_save_path)


def evaluate(config, val_iter, model, data_loader):
    model.eval()
    real_true, real_pred = [], []
    show = True
    with torch.no_grad():
        for idx, (sen, token_ids, labels) in enumerate(val_iter):
            token_ids = token_ids.to(config.device)
            labels = labels.to(config.device)
            padding_mask = (token_ids == data_loader.PAD_IDX).transpose(0, 1)
            logits = model(input_ids=token_ids,  # [src_len, batch_size]
                           attention_mask=padding_mask,  # [batch_size,src_len]
                           token_type_ids=None,
                           position_ids=None,
                           labels=None)  # [src_len, batch_size]
            # logits :[src_len, batch_size, num_labels]
            if show:
                show_result(sen[:10], logits[:, :10], token_ids[:, :10], config.entities)
                show = False
            _, t, p = accuracy(logits, labels, config.ignore_idx)
            real_true += t
            real_pred += p
    model.train()
    target_names = list(config.entities.keys())
    logging.info(f"\n{classification_report(real_true, real_pred, target_names=target_names)}")
    return accuracy_score(real_true, real_pred)


def get_ner_tags(logits, token_ids, entities, SEP_IDX=102):
    """
    :param logits:  [src_len,batch_size,num_samples]
    :param token_ids: # [src_len,batch_size]
    :return:
    e.g.
    logits = torch.tensor([[[0.4, 0.7, 0.2],[0.5, 0.4, 0.1],[0.1, 0.2, 0.3],[0.5, 0.7, 0.2],[0.1, 0.2, 0.5]],
                       [[0.3, 0.2, 0.5],[0.7, 0.8, 0.4],[0.1, 0.1, 0.3],[0.9, 0.2, 0.1],[0.1, 0.5,0.2]]])
    logits = logits.transpose(0, 1)  # [src_len,batch_size,num_samples]
    token_ids = torch.tensor([[101, 2769, 511, 102, 0],
                              [101, 56, 33, 22, 102]]).transpose(0, 1)  # [src_len,batch_size]
    labels, probs = get_ner_tags(logits, token_ids, entities)
    [['O', 'B-LOC'], ['B-ORG', 'B-LOC', 'O']]
    [[0.5, 0.30000001192092896], [0.800000011920929, 0.30000001192092896, 0.8999999761581421]]
    """
    # entities = {'O': 0, 'B-ORG': 1, 'B-LOC': 2, 'B-PER': 3, 'I-ORG': 4, 'I-LOC': 5, 'I-PER': 6}
    label_list = list(entities.keys())
    logits = logits[1:].transpose(0, 1)  # [batch_size,src_len-1,num_samples]
    prob, y_pred = torch.max(logits, dim=-1)  # prob, y_pred: [batch_size,src_len-1]
    token_ids = token_ids[1:].transpose(0, 1)  # [ batch_size,src_len-1]， 去掉[cls]
    assert y_pred.shape == token_ids.shape
    labels = []
    probs = []
    for sample in zip(y_pred, token_ids, prob):
        tmp_label, tmp_prob = [], []
        for item in zip(*sample):
            if item[1] == SEP_IDX:  # 忽略最后一个[SEP]字符
                break
            tmp_label.append(label_list[item[0]])
            tmp_prob.append(item[2].item())
        labels.append(tmp_label)
        probs.append(tmp_prob)
    return labels, probs


def pretty_print(sentences, labels, entities):
    """
    :param sentences:
    :param labels:
    :param entities:
    :return:
    e.g.
    labels = [['B-PER','I-PER', 'O','O','O','O','O','O','O','O','O','O','B-LOC','I-LOC','B-LOC','I-LOC','O','O','O','O'],
    ['B-LOC','I-LOC','O','B-LOC','I-LOC','O','B-LOC','I-LOC','I-LOC','O','B-LOC','I-LOC','O','O','O','B-PER','I-PER','O','O','O','O','O','O']]
    sentences=["涂伊说，如果有机会他想去赤壁看一看！",
               "丽江、大理、九寨沟、黄龙等都是涂伊想去的地方！"]
    entities = {'O': 0, 'B-ORG': 1, 'B-LOC': 2, 'B-PER': 3, 'I-ORG': 4, 'I-LOC': 5, 'I-PER': 6}


    句子：涂伊说，如果有机会他想去黄州赤壁看一看！
    涂伊:  PER
    黄州:  LOC
    赤壁:  LOC
    句子：丽江、大理、九寨沟、黄龙等都是涂伊想去的地方！
    丽江:  LOC
    大理:  LOC
    九寨沟:  LOC
    黄龙:  LOC
    涂伊:  PER
    """

    sep_tag = [tag for tag in list(entities.keys()) if 'I' not in tag]
    result = []
    for sen, label in zip(sentences, labels):
        logging.info(f"句子：{sen}")
        last_tag = None
        for item in zip(sen + "O", label + ['O']):
            if item[1] in sep_tag:  #
                if len(result) > 0:
                    entity = "".join(result)
                    logging.info(f"\t{entity}:  {last_tag.split('-')[-1]}")
                    result = []
                if item[1] != 'O':
                    result.append(item[0])
                    last_tag = item[1]
            else:
                result.append(item[0])
                last_tag = item[1]


def show_result(sentences, logits, token_ids, entities):
    labels, _ = get_ner_tags(logits, token_ids, entities)
    pretty_print(sentences, labels, entities)


def inference(config, sentences=None):
    model = BertForTokenClassification(config,
                                       config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir,
                                   config.model_save_name)
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    else:
        raise ValueError(f" 本地模型{model_save_path}不存在，请先训练模型。")
    model = model.to(config.device)
    data_loader = LoadChineseNERDataset(
        entities=config.entities,
        num_labels=config.num_labels,
        ignore_idx=config.ignore_idx,
        vocab_path=config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(
            config.pretrained_model_dir).tokenize,
        batch_size=config.batch_size,
        max_sen_len=config.max_sen_len,
        split_sep=config.split_sep,
        max_position_embeddings=config.max_position_embeddings,
        pad_index=config.pad_token_id,
        is_sample_shuffle=config.is_sample_shuffle)
    _, token_ids, _ = data_loader.make_inference_samples(sentences)
    token_ids = token_ids.to(config.device)
    padding_mask = (token_ids == data_loader.PAD_IDX).transpose(0, 1)
    logits = model(input_ids=token_ids,  # [src_len, batch_size]
                   attention_mask=padding_mask)  # [batch_size,src_len]
    show_result(sentences, logits, token_ids, config.entities)


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    sentences = ['智光拿出石壁拓文为乔峰详述事情始末，乔峰方知自己原本姓萧，乃契丹后族。',
                 '当乔峰问及带头大哥时，却发现智光大师已圆寂。',
                 '乔峰、阿朱相约找最后知情人康敏问完此事后，就到塞外骑马牧羊，再不回来。']
    inference(config, sentences)
