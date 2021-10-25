import sys

sys.path.append('../')
from model.BertForClassification.BertForSentenceClassification import BertForSentenceClassification
from model.BasicBert.BertConfig import BertConfig

json_file = '../bert_base_chinese/config.json'
config = BertConfig.from_json_file(json_file)
model = BertForSentenceClassification(config, 10)
print(model)
