import sys

sys.path.append('../')
from Tasks.TaskForSingleSentenceClassification import ModelConfig
from utils.data_helpers import LoadSingleSentenceClassificationDataset
from transformers import BertTokenizer

if __name__ == '__main__':
    model_config = ModelConfig()
    load_dataset = LoadSingleSentenceClassificationDataset(
        vocab_path=model_config.vocab_path,
        tokenizer=BertTokenizer.from_pretrained(model_config.pretrained_model_dir).tokenize,
        batch_size=model_config.batch_size,
        max_sen_len=model_config.max_sen_len,
        split_sep=model_config.split_sep,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_index=model_config.pad_token_id,
        is_sample_shuffle=model_config.is_sample_shuffle)

    train_iter, test_iter, val_iter = \
        load_dataset.load_train_val_test_data(model_config.train_file_path,
                                              model_config.val_file_path,
                                              model_config.test_file_path)
    for sample, label in train_iter:
        print(sample.shape)  # [seq_len,batch_size]
        print(sample.transpose(0, 1))
        padding_mask = (sample == load_dataset.PAD_IDX).transpose(0, 1)
        print(padding_mask)
        # print(label)
        break