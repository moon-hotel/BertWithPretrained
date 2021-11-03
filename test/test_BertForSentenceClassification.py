import sys

sys.path.append('../')
from model.DownstreamTasks.BertForSentenceClassification import BertForSentenceClassification
from model.BasicBert.BertConfig import BertConfig
import torch

json_file = '../bert_base_chinese/config.json'
config = BertConfig.from_json_file(json_file)
config.__dict__['num_labels'] = 10
model = BertForSentenceClassification(config)

input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
attention_mask = torch.tensor([[1, 0], [1, 0], [1, 1], [1, 1]])
logits = model(input_ids=input_ids,
               attention_mask=attention_mask)
print(logits.shape)
