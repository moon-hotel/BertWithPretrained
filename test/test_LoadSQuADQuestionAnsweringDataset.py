import sys

sys.path.append('../')
from utils.data_helpers import LoadSQuADQuestionAnsweringDataset
from transformers import BertTokenizer
from Tasks.TaskForSQuADQuestionAnswering import ModelConfig

if __name__ == '__main__':
    model_config = ModelConfig()
    data_loader = LoadSQuADQuestionAnsweringDataset(vocab_path=model_config.vocab_path,
                                                    tokenizer=BertTokenizer.from_pretrained(
                                                        model_config.pretrained_model_dir).tokenize,
                                                    batch_size=3,
                                                    max_sen_len=512,
                                                    max_position_embeddings=512,
                                                    pad_index=0,
                                                    is_sample_shuffle=True,
                                                    )

    # examples = data_loader.preprocessing(model_config.train_file_path)
    # print(examples)
    train_iter, test_iter, val_iter = data_loader. \
        load_train_val_test_data(test_file_path=model_config.test_file_path,
                                 train_file_path=model_config.train_file_path,
                                 only_test=False)

    for batch_input, batch_seg, batch_label in train_iter:
        print("=====================>")
        print(f"intput_ids shape: {batch_input.shape}")  # [max_len, batch_size]
        print(batch_input.transpose(0, 1).tolist())
        print(f"token_type_ids shape: {batch_seg.shape}")  # [max_len, batch_size]
        print(batch_seg.transpose(0, 1).tolist())
        print(batch_label)  # [batch_size,2]

        for i in range(batch_input.size(-1)):
            sample = batch_input.transpose(0, 1)[i]
            start_pos, end_pos = batch_label[i][0], batch_label[i][1]
            strs = [data_loader.vocab.itos[s] for s in sample]  # 原始tokens
            answer = " ".join(strs[start_pos:(end_pos + 1)]).replace(" ##", "")
            strs = " ".join(strs).replace(" ##", "").split('[SEP]')
            question, context = strs[0], strs[1]
            print(f"问题：{question}")
            print(f"描述：{context}")
            print(f"正确答案：{answer}")
            print(f"答案起止：{start_pos, end_pos}")
