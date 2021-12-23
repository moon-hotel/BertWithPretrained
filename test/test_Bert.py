import sys

sys.path.append('../')
from model.BasicBert.BertEmbedding import BertEmbeddings
from model.BasicBert.Bert import BertAttention
from model.BasicBert.Bert import BertLayer
from model.BasicBert.Bert import BertEncoder
from model.BasicBert.Bert import BertModel
from model.BasicBert.BertConfig import BertConfig
import torch
from utils.log_helper import logger_init

if __name__ == '__main__':
    logger_init()
    json_file = '../bert_base_chinese/config.json'
    config = BertConfig.from_json_file(json_file)
    config.__dict__['use_torch_multi_head'] = True  # 表示使用 torch框架中的MultiHeadAttention 注意力实现方法
    config.max_position_embeddings = 518 # 测试大于512时的情况
    src = torch.tensor([[1, 3, 5, 7, 9, 2, 3], [2, 4, 6, 8, 10, 0, 0]], dtype=torch.long)
    src = src.transpose(0, 1)  # [src_len, batch_size]
    print(f"input shape [src_len,batch_size]: ", src.shape)
    token_type_ids = torch.LongTensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]]).transpose(0, 1)
    attention_mask = torch.tensor([[True, True, True, True, True, True, True],
                                   [True, True, True, True, True, False, False]])
    # attention_mask 实际就是Transformer中指代的key_padding_mask

    # ------ BertEmbedding -------
    bert_embedding = BertEmbeddings(config)
    bert_embedding_result = bert_embedding(src, token_type_ids=token_type_ids)
    # [src_len, batch_size, hidden_size]

    # 测试类BertAttention
    bert_attention = BertAttention(config)
    bert_attention_output = bert_attention(bert_embedding_result, attention_mask=attention_mask)
    print(f"BertAttention output shape [src_len, batch_size, hidden_size]: ", bert_attention_output.shape)

    # 测试类BertLayer
    bert_layer = BertLayer(config)
    bert_layer_output = bert_layer(bert_embedding_result, attention_mask)
    print(f"BertLayer output shape [src_len, batch_size, hidden_size]: ", bert_layer_output.shape)

    # 测试类BertEncoder
    bert_encoder = BertEncoder(config)
    bert_encoder_outputs = bert_encoder(bert_embedding_result, attention_mask)
    print(f"num of BertEncoder [config.num_hidden_layers]: ", len(bert_encoder_outputs))
    print(f"each output shape in BertEncoder [src_len, batch_size, hidden_size]: ", bert_encoder_outputs[0].shape)

    # 测试类BertModel
    position_ids = torch.arange(src.size()[0]).expand((1, -1))  # [1,src_len]
    bert_model = BertModel(config)
    bert_model_output = bert_model(input_ids=src,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids)[0]
    print(f"BertModel's pooler output shape [batch_size, hidden_size]: ", bert_model_output.shape)
    print("\n  =======  BertMolde 参数: ========")
    for param_tensor in bert_model.state_dict():
        print(param_tensor, "\t", bert_model.state_dict()[param_tensor].size())

    print(f"\n  =======  测试BertModel载入预训练模型： ========")
    model = BertModel.from_pretrained(config, pretrained_model_dir="../bert_base_chinese")
