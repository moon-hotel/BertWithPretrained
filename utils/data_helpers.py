import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
import collections
import six


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


def cache(func):
    """
    本修饰器的作用是将SQuAD数据集中data_process()方法处理后的结果进行缓存，下次使用时可直接载入！
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
        :param is_sample_shuffle: 是否打乱训练集样本（只针对训练集）
                在后续构造DataLoader时，验证集和测试集均指定为了固定顺序（即不进行打乱），修改程序时请勿进行打乱
                因为当shuffle为True时，每次通过for循环遍历data_iter时样本的顺序都不一样，这会导致在模型预测时
                返回的标签顺序与原始的顺序不一样，不方便处理。

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

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        test_data, _ = self.data_process(test_file_path)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(train_file_path)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(val_file_path)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
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
        batch_mask = (batch_qa == self.PAD_IDX).view(
            [-1, self.num_choice, batch_qa.size(-1)])
        # reshape 至 [batch_size, num_choice, max_len]
        batch_qa = batch_qa.view([-1, self.num_choice, batch_qa.size(-1)])
        batch_seg = pad_sequence(batch_seg,  # [batch_size*num_choice,max_len]
                                 padding_value=self.PAD_IDX,
                                 batch_first=True,
                                 max_len=self.max_sen_len)
        # reshape 至 [batch_size, num_choice, max_len]
        batch_seg = batch_seg.view([-1, self.num_choice, batch_seg.size(-1)])
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_qa, batch_seg, batch_mask, batch_label


class LoadSQuADQuestionAnsweringDataset(LoadSingleSentenceClassificationDataset):
    """
    Args:
        doc_stride: When splitting up a long document into chunks, how much stride to
                    take between chunks.
                    当上下文过长时，按滑动窗口进行移动，doc_stride表示每次移动的距离
        max_query_length: The maximum number of tokens for the question. Questions longer than
                    this will be truncated to this length.
                    限定问题的最大长度，过长时截断
        n_best_size: 对预测出的答案近后处理时，选取的候选答案数量
        max_answer_length: 在对候选进行筛选时，对答案最大长度的限制

    """

    def __init__(self, doc_stride=64,
                 max_query_length=64,
                 n_best_size=20,
                 max_answer_length=30,
                 **kwargs):
        super(LoadSQuADQuestionAnsweringDataset, self).__init__(**kwargs)
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length

    @staticmethod
    def get_format_text_and_word_offset(text):
        """
        格式化原始输入的文本（去除多个空格）,同时得到每个字符所属的元素（单词）的位置
        这样，根据原始数据集中所给出的起始index(answer_start)就能立马判定它在列表中的位置。
        :param text:
        :return:
        e.g.
            text = "Architecturally, the school has a Catholic character. "
            return:['Architecturally,', 'the', 'school', 'has', 'a', 'Catholic', 'character.'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3,
             3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        """

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        # 以下这个for循环的作用就是将原始context中的内容进行格式化
        for c in text:  # 遍历paragraph中的每个字符
            if is_whitespace(c):  # 判断当前字符是否为空格（各类空格）
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:  # 如果前一个字符是空格
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c  # 在list的最后一个元素中继续追加字符
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return doc_tokens, char_to_word_offset

    def preprocessing(self, filepath, is_training=True):
        """
        将原始数据进行预处理，同时返回得到答案在原始context中的具体开始和结束位置（以单词为单位）
        :param filepath:
        :param is_training:
        :return:
        返回形式为一个二维列表，内层列表中的各个元素分别为 ['问题ID','原始问题文本','答案文本','context文本',
        '答案在context中的开始位置','答案在context中的结束位置']，并且二维列表中的一个元素称之为一个example,即一个example由六部分组成
        如下示例所示：
        [['5733be284776f41900661182', 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
        'Saint Bernadette Soubirous', 'Architecturally, the school has a Catholic character......',
        90, 92],
         ['5733be284776f4190066117f', ....]]
        """
        with open(filepath, 'r') as f:
            raw_data = json.loads(f.read())
            data = raw_data['data']
        examples = []
        for i in tqdm(range(len(data)), ncols=80, desc="正在遍历每一个段落"):  # 遍历每一个paragraphs
            paragraphs = data[i]['paragraphs']  # 取第i个paragraphs
            for j in range(len(paragraphs)):  # 遍历第i个paragraphs的每个context
                context = paragraphs[j]['context']  # 取第j个context
                context_tokens, word_offset = self.get_format_text_and_word_offset(context)
                qas = paragraphs[j]['qas']  # 取第j个context下的所有 问题-答案 对
                for k in range(len(qas)):  # 遍历第j个context中的多个 问题-答案 对
                    question_text = qas[k]['question']
                    qas_id = qas[k]['id']
                    if is_training:
                        answer_offset = qas[k]['answers'][0]['answer_start']
                        orig_answer_text = qas[k]['answers'][0]['text']
                        answer_length = len(orig_answer_text)
                        start_position = word_offset[answer_offset]
                        end_position = word_offset[answer_offset + answer_length - 1]
                        actual_text = " ".join(
                            context_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(orig_answer_text.strip().split())
                        if actual_text.find(cleaned_answer_text) == -1:
                            logging.warning("Could not find answer: '%s' vs. '%s'",
                                            actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                    examples.append([qas_id, question_text, orig_answer_text,
                                     " ".join(context_tokens), start_position, end_position])
        return examples

    @staticmethod
    def improve_answer_span(context_tokens,
                            answer_tokens,
                            start_position,
                            end_position):
        """
        本方法的作用有两个：
            1. 如https://github.com/google-research/bert中run_squad.py里的_improve_answer_span()函数一样，
               用于提取得到更加匹配答案的起止位置；
            2. 根据原始起止位置，提取得到token id中答案的起止位置
        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.

        context = "The leader was John Smith (1895-1943).
        answer_text = "1985"
        :param context_tokens: ['the', 'leader', 'was', 'john', 'smith', '(', '1895', '-', '1943', ')', '.']
        :param answer_tokens: ['1895']
        :param start_position: 5
        :param end_position: 5
        :return: [6,6]
        再例如：
        context = "Virgin mary reputedly appeared to Saint Bernadette Soubirous in 1858"
        answer_text = "Saint Bernadette Soubirous"
        :param context_tokens: ['virgin', 'mary', 'reputed', '##ly', 'appeared', 'to', 'saint', 'bern', '##ade',
                                '##tte', 'so', '##ub', '##iro', '##us', 'in', '1858']
        :param answer_tokens: ['saint', 'bern', '##ade', '##tte', 'so', '##ub', '##iro', '##us'
        :param start_position = 5
        :param end_position = 7
        return (6,13)

        """
        new_end = None
        for i in range(start_position, len(context_tokens)):
            if context_tokens[i] != answer_tokens[0]:
                continue
            for j in range(len(answer_tokens)):
                if answer_tokens[j] != context_tokens[i + j]:
                    break
                new_end = i + j
            if new_end - i + 1 == len(answer_tokens):
                return i, new_end
        return start_position, end_position

    @staticmethod
    def get_token_to_orig_map(input_tokens, origin_context, tokenizer):
        """
           本函数的作用是根据input_tokens和原始的上下文，返回得input_tokens中每个单词在原始单词中所对应的位置索引
           :param input_tokens:  ['[CLS]', 'to', 'whom', 'did', 'the', 'virgin', '[SEP]', 'architectural', '##ly',
                                   ',', 'the', 'school', 'has', 'a', 'catholic', 'character', '.', '[SEP']
           :param origin_context: "Architecturally, the Architecturally, test, Architecturally,
                                    the school has a Catholic character. Welcome moon hotel"
           :param tokenizer:
           :return: {7: 4, 8: 4, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 10}
                   含义是input_tokens[7]为origin_context中的第4个单词 Architecturally,
                        input_tokens[8]为origin_context中的第4个单词 Architecturally,
                        ...
                        input_tokens[10]为origin_context中的第5个单词 the
           """
        origin_context_tokens = origin_context.split()
        token_id = []
        str_origin_context = ""
        for i in range(len(origin_context_tokens)):
            tokens = tokenizer(origin_context_tokens[i])
            str_token = "".join(tokens)
            str_origin_context += "" + str_token
            for _ in str_token:
                token_id.append(i)

        key_start = input_tokens.index('[SEP]') + 1
        tokenized_tokens = input_tokens[key_start:-1]
        str_tokenized_tokens = "".join(tokenized_tokens)
        index = str_origin_context.index(str_tokenized_tokens)
        value_start = token_id[index]
        token_to_orig_map = {}
        # 处理这样的边界情况： Building's gold   《==》   's', 'gold', 'dome'
        token = tokenizer(origin_context_tokens[value_start])
        for i in range(len(token), -1, -1):
            s1 = "".join(token[-i:])
            s2 = "".join(tokenized_tokens[:i])
            if s1 == s2:
                token = token[-i:]
                break

        while True:
            for j in range(len(token)):
                token_to_orig_map[key_start] = value_start
                key_start += 1
                if len(token_to_orig_map) == len(tokenized_tokens):
                    return token_to_orig_map
            value_start += 1
            token = tokenizer(origin_context_tokens[value_start])

    @cache
    def data_process(self, filepath, is_training=False, postfix='cache'):
        """

        :param filepath:
        :param is_training:
        :return: [[example_id, feature_id, input_ids, seg, start_position,
                    end_position, answer_text, example[0]],input_tokens,token_to_orig_map [],[],[]...]
                  分别对应：[原始样本Id,训练特征id,input_ids，seg，开始，结束，答案文本，问题id,input_tokens,token_to_orig_map]
        """
        logging.info(f"## 使用窗口滑动滑动，doc_stride = {self.doc_stride}")
        examples = self.preprocessing(filepath, is_training)
        all_data = []
        example_id, feature_id = 0, 1000000000
        # 由于采用了滑动窗口，所以一个example可能构造得到多个训练样本（即这里被称为feature）；
        # 因此，需要对其分别进行编号，并且这主要是用在预测后的结果后处理当中，训练时用不到
        # 当然，这里只使用feature_id即可，因为每个example其实对应的就是一个问题，所以问题ID和example_id本质上是一样的
        for example in tqdm(examples, ncols=80, desc="正在遍历每个问题（样本）"):
            question_tokens = self.tokenizer(example[1])
            if len(question_tokens) > self.max_query_length:  # 问题过长进行截取
                question_tokens = question_tokens[:self.max_query_length]
            question_ids = [self.vocab[token] for token in question_tokens]
            question_ids = [self.CLS_IDX] + question_ids + [self.SEP_IDX]
            context_tokens = self.tokenizer(example[3])
            context_ids = [self.vocab[token] for token in context_tokens]
            logging.debug(f"<<<<<<<<  进入新的example  >>>>>>>>>")
            logging.debug(f"## 正在预处理数据 {__name__} is_training = {is_training}")
            logging.debug(f"## 问题 id: {example[0]}")
            logging.debug(f"## 原始问题 text: {example[1]}")
            logging.debug(f"## 原始描述 text: {example[3]}")
            start_position, end_position, answer_text = -1, -1, None
            if is_training:
                start_position, end_position = example[4], example[5]
                answer_text = example[2]
                answer_tokens = self.tokenizer(answer_text)
                start_position, end_position = self.improve_answer_span(context_tokens,
                                                                        answer_tokens,
                                                                        start_position,
                                                                        end_position)
            rest_len = self.max_sen_len - len(question_ids) - 1
            context_ids_len = len(context_ids)
            logging.debug(f"## 上下文长度为：{context_ids_len}, 剩余长度 rest_len 为 ： {rest_len}")
            if context_ids_len > rest_len:  # 长度超过max_sen_len,需要进行滑动窗口
                logging.debug(f"## 进入滑动窗口 …… ")
                s_idx, e_idx = 0, rest_len
                while True:
                    # We can have documents that are longer than the maximum sequence length.
                    # To deal with this we do a sliding window approach, where we take chunks
                    # of the up to our max length with a stride of `doc_stride`.
                    tmp_context_ids = context_ids[s_idx:e_idx]
                    tmp_context_tokens = [self.vocab.itos[item] for item in tmp_context_ids]
                    logging.debug(f"## 滑动窗口范围：{s_idx, e_idx},example_id: {example_id}, feature_id: {feature_id}")
                    # logging.debug(f"## 滑动窗口取值：{tmp_context_tokens}")
                    input_ids = torch.tensor(question_ids + tmp_context_ids + [self.SEP_IDX])
                    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + tmp_context_tokens + ['[SEP]']
                    seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                    seg = torch.tensor(seg)
                    if is_training:
                        new_start_position, new_end_position = 0, 0
                        if start_position >= s_idx and end_position <= e_idx:  # in train
                            logging.debug(f"## 滑动窗口中存在答案 -----> ")
                            new_start_position = start_position - s_idx
                            new_end_position = new_start_position + (end_position - start_position)

                            new_start_position += len(question_ids)
                            new_end_position += len(question_ids)
                            logging.debug(f"## 原始答案：{answer_text} <===>处理后的答案："
                                          f"{' '.join(input_tokens[new_start_position:(new_end_position + 1)])}")
                        all_data.append([example_id, feature_id, input_ids, seg, new_start_position,
                                         new_end_position, answer_text, example[0], input_tokens])
                        logging.debug(f"## start pos:{new_start_position}")
                        logging.debug(f"## end pos:{new_end_position}")
                    else:
                        all_data.append([example_id, feature_id, input_ids, seg, start_position,
                                         end_position, answer_text, example[0], input_tokens])
                        logging.debug(f"## start pos:{start_position}")
                        logging.debug(f"## end pos:{end_position}")
                    token_to_orig_map = self.get_token_to_orig_map(input_tokens, example[3], self.tokenizer)
                    all_data[-1].append(token_to_orig_map)
                    logging.debug(f"## example id: {example_id}")
                    logging.debug(f"## feature id: {feature_id}")
                    logging.debug(f"## input_tokens: {input_tokens}")
                    logging.debug(f"## input_ids:{input_ids.tolist()}")
                    logging.debug(f"## segment ids:{seg.tolist()}")
                    logging.debug("======================\n")
                    feature_id += 1
                    if e_idx >= context_ids_len:
                        break
                    s_idx += self.doc_stride
                    e_idx += self.doc_stride

            else:
                input_ids = torch.tensor(question_ids + context_ids + [self.SEP_IDX])
                input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
                seg = [0] * len(question_ids) + [1] * (len(input_ids) - len(question_ids))
                seg = torch.tensor(seg)
                if is_training:
                    start_position += (len(question_ids))
                    end_position += (len(question_ids))
                token_to_orig_map = self.get_token_to_orig_map(input_tokens, example[3], self.tokenizer)
                all_data.append([example_id, feature_id, input_ids, seg, start_position,
                                 end_position, answer_text, example[0], input_tokens, token_to_orig_map])
                logging.debug(f"## input_tokens: {input_tokens}")
                logging.debug(f"## input_ids:{input_ids.tolist()}")
                logging.debug(f"## segment ids:{seg.tolist()}")
                logging.debug("======================\n")
                feature_id += 1
            example_id += 1
        #  all_data[0]: [原始样本Id,训练特征id,input_ids，seg，开始，结束，答案文本，问题id, input_tokens,ori_map]
        data = {'all_data': all_data, 'max_len': self.max_sen_len, 'examples': examples}
        return data

    def generate_batch(self, data_batch):
        batch_input, batch_seg, batch_label, batch_qid = [], [], [], []
        batch_example_id, batch_feature_id, batch_map = [], [], []
        for item in data_batch:
            # item: [原始样本Id,训练特征id,input_ids，seg，开始，结束，答案文本，问题id,input_tokens,ori_map]
            batch_example_id.append(item[0])  # 原始样本Id
            batch_feature_id.append(item[1])  # 训练特征id
            batch_input.append(item[2])  # input_ids
            batch_seg.append(item[3])  # seg
            batch_label.append([item[4], item[5]])  # 开始, 结束
            batch_qid.append(item[7])  # 问题id
            batch_map.append(item[9])  # ori_map

        batch_input = pad_sequence(batch_input,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=False,
                                   max_len=self.max_sen_len)  # [max_len,batch_size]
        batch_seg = pad_sequence(batch_seg,  # [batch_size,max_len]
                                 padding_value=self.PAD_IDX,
                                 batch_first=False,
                                 max_len=self.max_sen_len)  # [max_len, batch_size]
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        # [max_len,batch_size] , [max_len, batch_size] , [batch_size,2], [batch_size,], [batch_size,]
        return batch_input, batch_seg, batch_label, batch_qid, batch_example_id, batch_feature_id, batch_map

    def load_train_val_test_data(self, train_file_path=None,
                                 val_file_path=None,
                                 test_file_path=None,
                                 only_test=True):
        doc_stride = str(self.doc_stride)
        max_sen_len = str(self.max_sen_len)
        max_query_length = str(self.max_query_length)
        postfix = doc_stride + '_' + max_sen_len + '_' + max_query_length
        data = self.data_process(filepath=test_file_path,
                                 is_training=False,
                                 postfix=postfix)
        test_data, examples = data['all_data'], data['examples']
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False,
                               collate_fn=self.generate_batch)
        if only_test:
            logging.info(f"## 成功返回测试集，一共包含样本{len(test_iter.dataset)}个")
            return test_iter, examples

        data = self.data_process(filepath=train_file_path,
                                 is_training=True,
                                 postfix=postfix)  # 得到处理好的所有样本
        train_data, max_sen_len = data['all_data'], data['max_len']
        _, val_data = train_test_split(train_data, test_size=0.3, random_state=2021)
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,  # 构造DataLoader
                              shuffle=False, collate_fn=self.generate_batch)
        logging.info(f"## 成功返回训练集样本（{len(train_iter.dataset)}）个、开发集样本（{len(val_iter.dataset)}）个"
                     f"测试集样本（{len(test_iter.dataset)}）个.")
        return train_iter, test_iter, val_iter

    @staticmethod
    def get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        # logits = [0.37203778 0.48594432 0.81051651 0.07998148 0.93529721 0.0476721
        #  0.15275263 0.98202781 0.07813079 0.85410559]
        # n_best_size = 4
        # return [7, 4, 9, 2]
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def get_final_text(self, pred_text, orig_text):
        """Project the tokenized prediction back to the original text."""

        # ref: https://github.com/google-research/bert/blob/master/run_squad.py
        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.

        tok_text = " ".join(self.tokenizer(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def write_prediction(self, test_iter, all_examples, logits_data, output_dir):
        """
        根据预测得到的logits将预测结果写入到本地文件中
        :param test_iter:
        :param all_examples:
        :param logits_data:
        :return:
        """
        qid_to_example_context = {}  # 根据qid取到其对应的context token
        for example in all_examples:
            context = example[3]
            context_list = context.split()
            qid_to_example_context[example[0]] = context_list
        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["text", "start_index", "end_index", "start_logit", "end_logit"])
        prelim_predictions = collections.defaultdict(list)
        for b_input, _, _, b_qid, _, _, b_map in tqdm(test_iter, ncols=80, desc="正在遍历候选答案"):
            # 取一个问题对应所有特征样本的预测logits（因为有了滑动窗口，所以原始一个context可以构造得到过个训练样本）
            all_logits = logits_data[b_qid[0]]
            for logits in all_logits:
                # 遍历每个logits的预测情况
                start_indexes = self.get_best_indexes(logits[1], self.n_best_size)
                # 得到开始位置几率最大的值对应的索引，例如可能是 [ 4,6,3,1]
                end_indexes = self.get_best_indexes(logits[2], self.n_best_size)
                # 得到结束位置几率最大的值对应的索引，例如可能是 [ 5,8,10,9]
                for start_index in start_indexes:
                    for end_index in end_indexes:  # 遍历所有存在的结果组合
                        if start_index >= b_input.size(0):
                            continue  # 起始索引大于token长度，忽略
                        if end_index >= b_input.size(0):
                            continue  # 结束索引大于token长度，忽略
                        if start_index not in b_map[0]:
                            continue  # 用来判断索引是否位于[SEP]之后的位置，因为答案只会在[SEP]以后出现
                        if end_index not in b_map[0]:
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            continue
                        token_ids = b_input.transpose(0, 1)[0]
                        strs = [self.vocab.itos[s] for s in token_ids]
                        tok_text = " ".join(strs[start_index:(end_index + 1)])
                        tok_text = tok_text.replace(" ##", "").replace("##", "")
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())

                        orig_doc_start = b_map[0][start_index]
                        orig_doc_end = b_map[0][end_index]
                        orig_tokens = qid_to_example_context[b_qid[0]][orig_doc_start:(orig_doc_end + 1)]
                        orig_text = " ".join(orig_tokens)
                        final_text = self.get_final_text(tok_text, orig_text)

                        prelim_predictions[b_qid[0]].append(_PrelimPrediction(
                            text=final_text,
                            start_index=int(start_index),
                            end_index=int(end_index),
                            start_logit=float(logits[1][start_index]),
                            end_logit=float(logits[2][end_index])))
                        # 此处为将每个qid对应的所有预测结果放到一起，因为一个qid对应的context应该滑动窗口
                        # 会有构造得到多个训练样本，而每个训练样本都会对应得到一个预测的logits
                        # 并且这里取了n_best个logits，所以组合后一个问题就会得到过个预测的答案

        for k, v in prelim_predictions.items():
            # 对每个qid对应的所有预测答案按照start_logit+end_logit的大小进行排序
            prelim_predictions[k] = sorted(prelim_predictions[k],
                                           key=lambda x: (x.start_logit + x.end_logit),
                                           reverse=True)
        best_results, all_n_best_results = {}, {}
        for k, v in prelim_predictions.items():
            best_results[k] = v[0].text  # 取最好的第一个结果
            all_n_best_results[k] = v  # 保存所有预测结果
        with open(os.path.join(output_dir, f"best_result.json"), 'w') as f:
            f.write(json.dumps(best_results, indent=4) + '\n')
        with open(os.path.join(output_dir, f"best_n_result.json"), 'w') as f:
            f.write(json.dumps(all_n_best_results, indent=4) + '\n')
