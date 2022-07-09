import sys

sys.path.append('../')
from model import BertForTokenClassification
from model import BertConfig
import torch

json_file = '../bert_base_chinese/config.json'
config = BertConfig.from_json_file(json_file)
config.__dict__['num_labels'] = 10
model = BertForTokenClassification(config)
src_len = 8
batch_size = 2
print(config.num_labels)
input_ids = torch.randint(0, 20, [src_len, batch_size])  # [src_len, batch_size
attention_mask = torch.tensor([[False, False, False, False, False, True, True, True],  # [batch_size,src_len]
                               [False, False, False, True, True, True, True, True]])
labels = torch.randint(0, config.num_labels, [src_len, batch_size])
loss, logits = model(input_ids=input_ids,
                     attention_mask=attention_mask,
                     labels=labels)
print(logits)
print(loss)
