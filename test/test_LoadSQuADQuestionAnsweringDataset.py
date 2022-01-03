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
                                                    batch_size=5,
                                                    max_sen_len=120,
                                                    max_position_embeddings=512,
                                                    pad_index=0,
                                                    is_sample_shuffle=False,
                                                    doc_stride=8,
                                                    max_query_length=5
                                                    )

    # examples = data_loader.preprocessing(model_config.train_file_path)
    # print(examples)

    # train_data = data_loader.data_process(filepath=model_config.train_file_path,
    #                                       is_training=False)  # 得到处理好的所有样本
    train_iter, test_iter, val_iter = data_loader. \
        load_train_val_test_data(test_file_path=model_config.test_file_path,
                                 train_file_path=model_config.train_file_path,
                                 only_test=False)

    for b_input, b_seg, b_label, b_qid, b_example_id, b_feature_id, b_map in train_iter:
        print("=====================>")
        print(f"intput_ids shape: {b_input.shape}")  # [max_len, batch_size]
        # print(b_input.transpose(0, 1).tolist())
        print(f"token_type_ids shape: {b_seg.shape}")  # [max_len, batch_size]
        print(b_seg.transpose(0, 1).tolist())
        print(b_label)  # [batch_size,2]
        print(b_map)  # [batch_size]

        for i in range(b_input.size(-1)):
            sample = b_input.transpose(0, 1)[i]
            start_pos, end_pos = b_label[i][0], b_label[i][1]
            strs = [data_loader.vocab.itos[s] for s in sample]  # 原始tokens
            answer = " ".join(strs[start_pos:(end_pos + 1)]).replace(" ##", "")
            strs = " ".join(strs).replace(" ##", "").split('[SEP]')
            question, context = strs[0], strs[1]
            print(f"问题ID：{b_qid[i]}")
            print(f"问题：{question}")
            # print(f"描述：{context}")
            print(f"正确答案：{answer}")
            print(f"答案起止：{start_pos, end_pos}")
            print(f"example ID：{b_example_id[i]}")
            print(f"feature ID：{b_feature_id[i]}")
