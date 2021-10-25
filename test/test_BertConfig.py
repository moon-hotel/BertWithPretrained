import sys

sys.path.append('../')
from model.BasicBert.BertConfig import BertConfig

if __name__ == '__main__':
    json_file = '../bert_base_chinese/config.json'
    config = BertConfig.from_json_file(json_file)
    print(config.hidden_size)
    position_embedding_type = getattr(config, "position_embedding_type", "absolute")
    print(position_embedding_type)