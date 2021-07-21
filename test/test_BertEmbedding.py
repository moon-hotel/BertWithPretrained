import sys

sys.path.append('../')
from model.BasicBert.BertEmbedding import TokenEmbedding
from model.BasicBert.BertEmbedding import PositionalEmbedding
from model.BasicBert.BertEmbedding import SegmentEmbedding
from model.BasicBert.BertEmbedding import BERTEmbedding
from model.BasicBert.BertConfig import Config
import torch

if __name__ == '__main__':
    src = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    src = src.transpose(0, 1)  # [src_len, batch_size]
    config = Config()
    print(f"input shape [src_len,batch_size]: ", src.shape)

    token_embedding = TokenEmbedding(vocab_size=config.vocab_size, emb_size=config.hidden_size)
    t_embedding = token_embedding(tokens=src)
    print(f"token embedding shape [src_len,batch_size,embed_size]: ", t_embedding.shape)

    pos_embedding = PositionalEmbedding(d_model=config.hidden_size)
    p_embedding = pos_embedding(x=src)
    print(f"pos embedding shape [src_len, 1, d_model]: ", p_embedding.shape)

    seg_embedding = SegmentEmbedding(embed_size=config.hidden_size)
    token_type_ids = torch.LongTensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]).transpose(0, 1)
    token_type_ids_embedding = seg_embedding(token_type_ids)
    print(f"seg embedding shape [src_len, batch_size, d_model]: ", token_type_ids_embedding.shape)

    bert_embedding = t_embedding + p_embedding + token_type_ids_embedding
    print(f"bert embedding shape [src_len, batch_size, d_model]: ", bert_embedding.shape)

    bert_embedding = BERTEmbedding(config)
    bert_embedding_result = bert_embedding(src, token_type_ids)
    print(f"bert embedding shape [src_len, batch_size, d_model]: ", bert_embedding_result.shape)
