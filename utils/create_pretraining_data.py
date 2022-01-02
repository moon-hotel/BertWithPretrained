import logging
import random
from tqdm import tqdm
from .data_helpers import build_vocab
from .data_helpers import pad_sequence
import torch
from torch.utils.data import DataLoader
import os


def read_wiki2(filepath=None):
    """
    本函数的作用是格式化原始的wikitext-2数据集
    下载地址为：https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    :param filepath:
    :return: 最终的返回形式为一个二维list，外层list中的每个元素为一个段落；内层list中每个元素为一个段落所有句子的集合。
            [ [sentence 1, sentence 2, ...], [sentence 1, sentence 2,...],...,[] ]
    该返回结果也是一个标准的格式，后续若需要载入其它数据集（包括中文），只需要首先将数据集处理成这样的格式；
    并在类LoadBertPretrainingDataset的get_format_data()方法中加入所实现的预处理函数即可完成整个预训练数据集的构造。
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()  # 一次读取所有行，每一行为一个段落
    # ①大写字母转换为小写字母
    # ②只取每一个段落中有至少两句话的段，因为后续要构造next Sentence
    paragraphs = []
    for line in tqdm(lines, ncols=80, desc=" ## 正在读取原始数据"):
        if len(line.split(' . ')) >= 2:
            paragraphs.append(line.strip().lower().split(' . '))
    random.shuffle(paragraphs)  # 将所有段落打乱
    return paragraphs


def read_custom(filepath=None):
    pass


def cache(func):
    """
    本修饰器的作用是将数据预处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadBertPretrainingDataset(object):
    r"""

    Arguments:

    """

    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=2021,
                 data_name='wiki2',
                 masked_rate=0.15,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        random.seed(random_state)

    def get_format_data(self, filepath):
        """
        本函数的作用是将数据集格式化成标准形式
        :param filepath:
        :return:  [ [sentence 1, sentence 2, ...], [sentence 1, sentence 2,...],...,[] ]
        """
        if self.data_name == 'wiki2':
            return read_wiki2(filepath)
        elif self.data_name == 'custom':
            return read_custom(filepath)
            # 在这里，可以调用你自己数据对应的格式化函数，
            # 但是返回格式需要同read_wiki2()保持一致。
        else:
            raise ValueError(f"数据 {self.data_name} 不存在对应的格式化函数，"
                             f"请参考函数 read_wiki(filepath) 实现对应的格式化函数！")

    @staticmethod
    def get_next_sentence_sample(sentence, next_sentence, paragraphs):
        """
        本函数的作用是根据给定的连续两句话和对应的段落，返回NSP任务中的句子对和标签
        :param sentence:  str
        :param next_sentence: str
        :param paragraphs: [str,str,...,]
        :return: sentence A, sentence B, True
        """
        if random.random() < 0.5:  # 产生[0,1)之间的一个随机数
            is_next = True
        else:
            # 这里random.choice的作用是从一个list中随机选出一个元素
            # ①先从所有段落中随机出一个段落；
            # ②再从随机出的一个段落中随机出一句话；
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions) >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_token_rate:  # 0.8
                masked_token_id = self.MASK_IDS
            else:
                # 10%的时间：保持词不变
                if random.random() < self.masked_token_unchanged_rate:  # 0.5
                    masked_token_id = token_ids[mlm_pred_position]
                # 10%的时间：用随机词替换该词
                else:
                    masked_token_id = random.randint(0, len(self.vocab.stoi))
            mlm_input_tokens_id[mlm_pred_position] = masked_token_id
            pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round(len(token_ids) * self.masked_rate))
        logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds)
        return mlm_input_tokens_id, mlm_label

    @cache
    def data_process(self, filepath, postfix='cache'):
        """
        本函数的作用是是根据格式化后的数据制作NSP和MLM两个任务对应的处理完成的数据
        :param filepath:
        :return:
        """
        paragraphs = self.get_format_data(filepath)
        # 返回的是一个二维列表，每个列表可以看做是一个段落（其中每个元素为一句话）
        data = []
        max_len = 0
        # 这里的max_len用来记录整个数据集中最长序列的长度，在后续可将其作为padding长度的标准
        for paragraph in tqdm(paragraphs, ncols=80, desc=" ## 正在构造NSP样本"):  # 遍历每个
            for i in range(len(paragraph) - 1):  # 遍历一个段落中的每一句话
                sentence, next_sentence, is_next = self.get_next_sentence_sample(
                    paragraph[i], paragraph[i + 1], paragraphs)  # 构造NSP样本
                logging.debug(f" ## 当前句文本：{sentence}")
                logging.debug(f" ## 下一句文本：{next_sentence}")
                logging.debug(f" ## 下一句标签：{is_next}")
                token_a_ids = [self.vocab[token] for token in self.tokenizer(sentence)]
                token_b_ids = [self.vocab[token] for token in self.tokenizer(next_sentence)]
                token_ids = [self.CLS_IDX] + token_a_ids + [self.SEP_IDX] + token_b_ids
                if len(token_ids) > self.max_position_embeddings - 1:
                    token_ids = token_ids[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
                token_ids += [self.SEP_IDX]
                logging.debug(f" ## 词元结果：{[self.vocab.itos[t] for t in token_ids]}")
                seg1 = [0] * (len(token_a_ids) + 2)  # 2 表示[CLS]和中间的[SEP]这两个字符
                seg2 = [1] * (len(token_ids) - len(seg1))
                segs = torch.tensor(seg1 + seg2, dtype=torch.long)
                logging.debug(f" ## Mask之前token ids:{token_ids}")
                logging.debug(f" ##      segment ids:{segs.tolist()},序列长度为 {len(segs)}")
                nsp_lable = torch.tensor(int(is_next), dtype=torch.long)
                mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids)
                token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
                mlm_label = torch.tensor(mlm_label, dtype=torch.long)
                max_len = max(max_len, token_ids.size(0))
                logging.debug(f" ## Mask之后token ids:{token_ids.tolist()}")
                logging.debug(f" ## Mask之后label ids:{mlm_label.tolist()}")
                logging.debug(f" ## 当前样本构造结束================== \n\n")
                data.append([token_ids, segs, nsp_lable, mlm_label])
        all_data = {'data': data, 'max_len': max_len}
        return all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_segs, b_nsp_label, b_mlm_label = [], [], [], []
        for (token_ids, segs, nsp_lable, mlm_label) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_segs.append(segs)
            b_nsp_label.append(nsp_lable)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_segs = pad_sequence(b_segs,  # [batch_size,max_len]
                              padding_value=self.PAD_IDX,
                              batch_first=True,
                              max_len=self.max_sen_len)
        # b_segs: [batch_size,src_len]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX).transpose(0, 1)
        # b_mask: [batch_size,max_len]

        b_nsp_label = torch.tensor(b_nsp_label, dtype=torch.long)
        # b_nsp_label: [batch_size]
        return b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label

    def load_train_val_test_data(self,
                                 train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        postfix = f"ml{self.max_sen_len}_rs{self.random_state}_mr{str(self.masked_rate)[2:]}" \
                  f"_mtr{str(self.masked_token_rate)[2:]}_mtur{str(self.masked_token_unchanged_rate)[2:]}"
        test_data = self.data_process(filepath=test_file_path,
                                      postfix=postfix)['data']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            logging.info(f"## 成功返回测试集，一共包含样本{len(test_iter.dataset)}个")
            return test_iter
        data = self.data_process(filepath=train_file_path, postfix=postfix)
        train_data, max_len = data['data'], data['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_data = self.data_process(filepath=val_file_path, postfix=postfix)['data']
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=self.generate_batch)
        logging.info(f"## 成功返回训练集样本（{len(train_iter.dataset)}）个、开发集样本（{len(val_iter.dataset)}）个"
                     f"测试集样本（{len(test_iter.dataset)}）个.")
        return train_iter, test_iter, val_iter
