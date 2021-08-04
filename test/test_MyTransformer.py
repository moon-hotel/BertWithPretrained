import sys

sys.path.append('../')
from model.BasicBert.MyTransformer import MyTransformer
import torch

if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    dmodel = 32
    tgt_len = 6
    num_head = 8
    src = torch.rand((src_len, batch_size, dmodel))  # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [batch_size, src_len]

    tgt = torch.rand((tgt_len, batch_size, dmodel))  # shape: [tgt_len, batch_size, embed_dim]
    tgt_key_padding_mask = torch.tensor([[True, True, True, False, False, False],
                                         [True, True, True, True, False, False]])  # shape: [batch_size, tgt_len]

    # ============ 测试 MyTransformer ============
    my_transformer = MyTransformer(d_model=dmodel, nhead=num_head, num_encoder_layers=6,
                                   num_decoder_layers=6, dim_feedforward=500)
    src_mask = my_transformer.generate_square_subsequent_mask(src_len)
    tgt_mask = my_transformer.generate_square_subsequent_mask(tgt_len)
    out = my_transformer(src=src, tgt=tgt, tgt_mask=tgt_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
    print(out.shape)
