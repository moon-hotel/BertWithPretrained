label_map = {'100': '0', '101': '1', '102': '2', '103': '3', '104': '4', '106': '5',
             '107': '6', '108': '7', '109': '8', '110': '9', '112': '10', '113': '11', '114': '12',
             '115': '13', '116': '14'}
import numpy as np


def format(path='./toutiao_cat_data.txt'):
    np.random.seed(2021)
    raw_data = open(path, 'r', encoding='utf-8').readlines()

    num_samples = len(raw_data)
    idx = np.random.permutation(num_samples)
    num_train, num_val = int(0.7 * num_samples), int(0.2 * num_samples)
    num_test = num_samples - num_train - num_val
    train_idx, val_idx, test_idx = idx[:num_train], idx[num_train:num_train + num_val], idx[-num_test:]
    f_train = open('./toutiao_train.txt', 'w', encoding='utf-8')
    f_val = open('./toutiao_val.txt', 'w', encoding='utf-8')
    f_test = open('./toutiao_test.txt', 'w', encoding='utf-8')

    for i in train_idx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = label_map[r[1]], r[3]
        f_train.write(text + '_!_' + label + '\n')
    f_train.close()

    for i in val_idx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = label_map[r[1]], r[3]
        f_val.write(text + '_!_' + label + '\n')
    f_val.close()

    for i in test_idx:
        r = raw_data[i].strip('\n').split('_!_')
        label, text = label_map[r[1]], r[3]
        f_test.write(text + '_!_' + label + '\n')
    f_test.close()


if __name__ == '__main__':
    format()
