import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd


class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class LoadSingleSentenceClassificationDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',  #
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):

        """

        :param vocab_path: 本地词表vocab.txt的路径
        :param tokenizer:
        :param batch_size:
        :param max_sen_len: 在对每个batch进行处理时的配置；
                            当max_sen_len = None时，即以每个batch中最长样本长度为标准，对其它进行padding
                            当max_sen_len = 'same'时，以整个数据集中最长样本为标准，对其它进行padding
                            当max_sen_len = 50， 表示以某个固定长度符样本进行padding，多余的截掉；
        :param split_sep: 文本和标签之前的分隔符，默认为'\t'
        :param max_position_embeddings: 指定最大样本长度，超过这个长度的部分将本截取掉
        :param is_sample_shuffle: 是否打乱样本

        """
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        # self.UNK_IDX = '[UNK]'

        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    def data_process(self, filepath):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param filepath: 数据集路径
        :return:
        """
        raw_iter = open(filepath, encoding="utf8").readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            s, l = line[0], line[1]
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_train_val_test_data(self, train_file_path, val_file_path, test_file_path):
        train_data, max_sen_len = self.data_process(train_file_path)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(val_file_path)
        test_data, _ = self.data_process(test_file_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


class LoadPairSentenceClassificationDataset(LoadSingleSentenceClassificationDataset):
    def __init__(self, **kwargs):
        super(LoadPairSentenceClassificationDataset, self).__init__(**kwargs)
        pass

    def data_process(self, filepath):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param filepath: 数据集路径
        :return:
        """
        raw_iter = open(filepath).readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split(self.split_sep)
            s1, s2, l = line[0], line[1], line[2]
            token1 = [self.vocab[token] for token in self.tokenizer(s1)]
            token2 = [self.vocab[token] for token in self.tokenizer(s2)]
            tmp = [self.CLS_IDX] + token1 + [self.SEP_IDX] + token2
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            seg1 = [0] * (len(token1) + 2)  # 2 表示[CLS]和中间的[SEP]这两个字符
            seg2 = [1] * (len(tmp) - len(seg1))
            segs = torch.tensor(seg1 + seg2, dtype=torch.long)
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, segs, l))
        return data, max_len

    def generate_batch(self, data_batch):
        batch_sentence, batch_seg, batch_label = [], [], []
        for (sen, seg, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_seg.append((seg))
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)  # [max_len,batch_size]
        batch_seg = pad_sequence(batch_seg,  # [batch_size,max_len]
                                 padding_value=self.PAD_IDX,
                                 batch_first=False,
                                 max_len=self.max_sen_len)  # [max_len, batch_size]
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_seg, batch_label


class LoadMultipleChoiceDataset(LoadSingleSentenceClassificationDataset):
    def __init__(self, num_choice=4, **kwargs):
        super(LoadMultipleChoiceDataset, self).__init__(**kwargs)
        self.num_choice = num_choice

    def data_process(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['startphrase']
        answers0, answers1 = data['ending0'], data['ending1']
        answers2, answers3 = data['ending2'], data['ending3']
        labels = [-1] * len(questions)
        if 'label' in data:  # 测试集中没有标签
            labels = data['label']
        all_data = []
        max_len = 0
        for i in tqdm(range(len(questions)), ncols=80):
            # 将问题中的每个word转换为字典中的token id
            t_q = [self.vocab[token] for token in self.tokenizer(questions[i])]
            t_q = [self.CLS_IDX] + t_q + [self.SEP_IDX]
            # 将答案中的每个word转换为字典中的token id
            t_a0 = [self.vocab[token] for token in self.tokenizer(answers0[i])]
            t_a1 = [self.vocab[token] for token in self.tokenizer(answers1[i])]
            t_a2 = [self.vocab[token] for token in self.tokenizer(answers2[i])]
            t_a3 = [self.vocab[token] for token in self.tokenizer(answers3[i])]
            # 计算最长序列的长度
            max_len = max(max_len, len(t_q) + max(len(t_a0), len(t_a1), len(t_a2), len(t_a3)))
            seg_q = [0] * len(t_q)
            # 加1表示还要加上问题和答案组合后最后一个[SEP]的长度
            seg_a0 = [1] * (len(t_a0) + 1)
            seg_a1 = [1] * (len(t_a1) + 1)
            seg_a2 = [1] * (len(t_a2) + 1)
            seg_a3 = [1] * (len(t_a3) + 1)
            all_data.append((t_q, t_a0, t_a1, t_a2, t_a3, seg_q,
                             seg_a0, seg_a1, seg_a2, seg_a3, labels[i]))
        return all_data, max_len

    def generate_batch(self, data_batch):
        batch_qa, batch_seg, batch_label = [], [], []

        def get_seq(q, a):
            seq = q + a
            if len(seq) > self.max_position_embeddings - 1:
                seq = seq[:self.max_position_embeddings - 1]
            return torch.tensor(seq + [self.SEP_IDX], dtype=torch.long)

        for item in data_batch:
            # 得到 每个问题组合其中一个答案的 input_ids 序列
            tmp_qa = [get_seq(item[0], item[1]),
                      get_seq(item[0], item[2]),
                      get_seq(item[0], item[3]),
                      get_seq(item[0], item[4])]
            # 得到 每个问题组合其中一个答案的 token_type_ids
            tmp_seg = [torch.tensor(item[5] + item[6], dtype=torch.long),
                       torch.tensor(item[5] + item[7], dtype=torch.long),
                       torch.tensor(item[5] + item[8], dtype=torch.long),
                       torch.tensor(item[5] + item[9], dtype=torch.long)]
            batch_qa.extend(tmp_qa)
            batch_seg.extend(tmp_seg)
            batch_label.append(item[-1])

        batch_qa = pad_sequence(batch_qa,  # [batch_size*num_choice,max_len]
                                padding_value=self.PAD_IDX,
                                batch_first=True,
                                max_len=self.max_sen_len)
        # reshape 至 [batch_size, num_choice, max_len]
        batch_qa = batch_qa.reshape([-1, self.num_choice, batch_qa.size(-1)])
        batch_seg = pad_sequence(batch_seg,  # [batch_size*num_choice,max_len]
                                 padding_value=self.PAD_IDX,
                                 batch_first=True,
                                 max_len=self.max_sen_len)
        # reshape 至 [batch_size, num_choice, max_len]
        batch_seg = batch_seg.reshape([-1, self.num_choice, batch_seg.size(-1)])
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_qa, batch_seg, batch_label
