import sys

sys.path.append('../')
from utils.data_helpers import LoadMultipleChoiceDataset
from Tasks.TaskForMultipleChoice import ModelConfig
from transformers import BertTokenizer


def trans_to_words(qas, itos):
    print("Question and Answer：")
    for s in qas:
        qa = " ".join(itos[idx] for idx in s)
        print(qa)


if __name__ == '__main__':
    model_config = ModelConfig()
    load_dataset = LoadMultipleChoiceDataset(vocab_path=model_config.vocab_path,
                                             tokenizer=BertTokenizer.from_pretrained(
                                                 model_config.pretrained_model_dir).tokenize,
                                             batch_size=2,
                                             max_sen_len=None,
                                             max_position_embeddings=512,
                                             pad_index=0,
                                             is_sample_shuffle=False,
                                             num_choice=model_config.num_labels)
    train_iter, test_iter, val_iter = \
        load_dataset.load_train_val_test_data(model_config.train_file_path,
                                              model_config.val_file_path,
                                              model_config.test_file_path)
    for qa, seg, mask, label in test_iter:
        print(" ### input ids:")
        print(qa.shape)  # [batch_size,num_choice, max_len]
        print(qa[0])
        print(" ### attention mask:")
        print(mask.shape)  # [batch_size,num_choice, max_len]
        print(mask[0])
        print(" ### token type ids:")
        print(seg.shape)  # [batch_size,num_choice, max_len]
        print(seg[0])
        print(label.shape)  # [batch_size]
        trans_to_words(qa[0], load_dataset.vocab.itos)
        break
