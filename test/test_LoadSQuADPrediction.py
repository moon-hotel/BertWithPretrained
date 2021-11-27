import sys

sys.path.append('../')
from utils.data_helpers import LoadSQuADQuestionAnsweringDataset
from transformers import BertTokenizer
from Tasks.TaskForSQuADQuestionAnswering import ModelConfig
from tqdm import tqdm
import json
import numpy as np

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

    test_iter = data_loader. \
        load_train_val_test_data(test_file_path=model_config.test_file_path,
                                 only_test=True)

    # 测试测试集输出结果
    for batch_input, batch_seg, batch_label, batch_qid in test_iter:
        print(batch_input.transpose(0, 1).tolist())
        print(batch_seg.transpose(0, 1).tolist())
        print(batch_label)
        print(batch_qid)

    y_start = [[45, 23], [78, 43, 88]]
    y_end = [[47, 25], [81, 57, 92]]
    y_pred = [np.hstack(y_start), np.hstack(y_end)]
    sample_id = 0
    results = {}
    for batch_input, _, _, batch_qid in tqdm(test_iter, ncols=80, desc="正在写入预测结果"):
        for i in range(batch_input.size(-1)):
            sample = batch_input.transpose(0, 1)[i]
            start_pos, end_pos = y_pred[0][sample_id], y_pred[1][sample_id]
            strs = [data_loader.vocab.itos[s] for s in sample]  # 原始tokens
            answer = " ".join(strs[start_pos:(end_pos + 1)]).replace(" ##", "")
            results[batch_qid[i]] = answer
            sample_id += 1
    with open(model_config.prediction_save_path, 'w') as f:
        results = json.dumps(results, ensure_ascii=False, indent=4)
        f.write(f"{results}")
