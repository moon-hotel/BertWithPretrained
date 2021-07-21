class Config:
    def __init__(self):
        self.attention_probs_dropout_prob = 0.1
        self.directionality = "bidi"
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.initializer_range = 0.02
        self.intermediate_size = 3072
        self.max_position_embeddings = 512
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.pooler_fc_size = 768
        self.pooler_num_attention_heads = 12
        self.pooler_num_fc_layers = 3
        self.pooler_size_per_head = 128
        self.pooler_type = "first_token_transform"
        self.type_vocab_size = 2
        self.vocab_size = 21128
