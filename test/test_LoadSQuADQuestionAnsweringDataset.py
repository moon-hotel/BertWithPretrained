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
                                                    batch_size=6,
                                                    max_sen_len=None,
                                                    max_position_embeddings=512,
                                                    pad_index=0,
                                                    is_sample_shuffle=False,
                                                    )
    # all_data, max_len= data_loader.data_process(model_config.train_file_path)
    test_iter = data_loader.load_train_val_test_data(test_file_path=model_config.test_file_path,
                                                     only_test=True)
    sample_id = 5
    for con_que, batch_seg, batch_label, raw_contexts in test_iter:
        print(f"context question shape: {con_que.shape}")
        print(con_que.transpose(0, 1))
        print(f"context question shape: {batch_seg.shape}")
        print(batch_seg.transpose(0, 1))
        print(batch_label.shape)
        context = con_que.transpose(0, 1)[sample_id]

        strs = " ".join([data_loader.vocab.itos[s] for s in context][1:])
        _, question, _ = strs.replace(" ##", "").split('[SEP]')

        start_pos, end_pos = batch_label[sample_id][0], batch_label[sample_id][1]
        raw_context = raw_contexts[sample_id]
        answer = " ".join(raw_context.split()[start_pos:end_pos + 1])
        print(f"context: \n {raw_context}")
        print(f"question:  {question}")
        print(f"answer:  {answer}")
        break
