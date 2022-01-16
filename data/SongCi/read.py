import json
import logging
from tqdm import tqdm
import os
import random


def format_data():
    """
    本函数的作用是格式化原始的ci.song.xxx.json数据集,将其保存分训练、验证和测试三部分
    :return:
    """

    def read_file(path=None):
        """
        读取每个json文件中的1000首词
        :param path:
        :return: 返回一个二维list
        """
        paras = []
        with open(path, encoding='utf-8') as f:
            data = json.loads(f.read())
            for item in data:
                tmp = item['paragraphs']
                if len(tmp) < 2:  # 小于两句的情况
                    continue
                if tmp[-1] == "词牌介绍":
                    tmp = tmp[:-2]
                paras.append(tmp)
        return paras

    def make_data(path, start, end):
        with open(path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(start, end, 1000), ncols=80, desc=" ## 正在制作训练数据"):
                path = f"ci.song.{i}.json"
                paragraphs = read_file(path)
                for para in paragraphs:
                    f.write("".join(para) + '\n')

    make_data('songci.train.txt', 0, 19001)  # 20 * 1000 首
    make_data('songci.valid.txt', 20000, 21001)  # 2 * 1000 首
    make_data('songci.test.txt', 20000, 21001)  # 2 * 1000 首


if __name__ == '__main__':
    format_data()
