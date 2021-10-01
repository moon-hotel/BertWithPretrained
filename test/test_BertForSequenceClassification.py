import sys

sys.path.append('../')
from model.BertForClassification.BertForSequenceClassification import BertForSequenceClassification
from model.BasicBert.BertConfig import BertConfig

json_file = '../pretrained_model/config.json'
config = BertConfig.from_json_file(json_file)
model = BertForSequenceClassification(config,10)
print(model)
