import sys

sys.path.append('../')
from model.DownstreamTasks.BertForSentenceClassification import BertForSentenceClassification
from model.BasicBert.BertConfig import BertConfig
import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    json_file = '../bert_base_chinese/config.json'
    config = BertConfig.from_json_file(json_file)
    config.__dict__['num_labels'] = 10
    config.__dict__['num_hidden_layers'] = 3
    model = BertForSentenceClassification(config)

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # [src_len,batch_size]
    attention_mask = torch.tensor([[1, 0], [1, 0], [1, 1], [1, 1]])  # [batch_size,src_len]
    logits = model(input_ids=input_ids,
                   attention_mask=attention_mask)
    print(logits.shape)
    writer = SummaryWriter('./runs')
    writer.add_graph(model, input_ids)
    # tensorboard 使用介绍见： https://mp.weixin.qq.com/s/8NB_hHYBq072xrSuCtgnEw
    #                       https://zhuanlan.zhihu.com/p/484289017

    # step1. cd BertWithPretrained/test/
    # step2. tensorboard --logdir=runs
    # step3. 打开浏览器输入: 127.0.0.1:6006打开查看
