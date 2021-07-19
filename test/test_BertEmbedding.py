import sys

sys.path.append('../')
from model.BasicBert.BertEmbedding import TokenEmbedding
from model.BasicBert.BertEmbedding import PositionalEmbedding
from model.BasicBert.BertEmbedding import SegmentEmbedding
from model.BasicBert.BertEmbedding import BERTEmbedding
import torch

if __name__ == '__main__':
    src = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    src = src.transpose(0, 1)  # [src_len, batch_size]
    vocab_size = 11
    emb_size = 512
    print(f"input shape [src_len,batch_size]: ", src.shape)

    token_embedding = TokenEmbedding(vocab_size=vocab_size, emb_size=emb_size)
    t_embedding = token_embedding(tokens=src)
    print(f"token embedding shape [src_len,batch_size,embed_size]: ", t_embedding.shape)

    pos_embedding = PositionalEmbedding(d_model=emb_size)
    p_embedding = pos_embedding(x=src)
    print(f"pos embedding shape [src_len, 1, d_model]: ", p_embedding.shape)

    seg_embedding = SegmentEmbedding(embed_size=emb_size)
    seg_label = torch.LongTensor([[1, 1, 1, 2, 2], [1, 1, 2, 2, 2]]).transpose(0, 1)
    s_embedding = seg_embedding(seg_label)
    print(f"seg embedding shape [src_len, batch_size, d_model]: ", s_embedding.shape)

    bert_embedding = t_embedding + p_embedding + s_embedding
    print(f"bert embedding shape [src_len, batch_size, d_model]: ", s_embedding.shape)

    bert_embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=emb_size)
    embedding_result = bert_embedding(src, seg_label)
    print(f"bert embedding shape [src_len, batch_size, d_model]: ", embedding_result.shape)
