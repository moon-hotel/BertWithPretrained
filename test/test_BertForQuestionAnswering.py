import sys

import torch

sys.path.append('../')
from model.DownstreamTasks.BertForQuestionAnswering import BertForQuestionAnswering
from model.BasicBert.BertConfig import BertConfig

json_file = '../bert_base_chinese/config.json'
config = BertConfig.from_json_file(json_file)
model = BertForQuestionAnswering(config)
input_ids = torch.randint(20, [20, 2])
attention_mask = torch.randint(2, size=[2, 20])
start_positions = torch.tensor([13, 1])
end_positions = torch.tensor([18, 21])
loss = model(input_ids=input_ids,
             attention_mask=attention_mask,
             start_positions=start_positions,
             end_positions=end_positions)
print(loss)
