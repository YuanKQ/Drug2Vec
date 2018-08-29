# -*- coding: utf-8 -*-
"""
__title__ = 'data_process.py'
__IDE__ = 'PyCharm'
__author__ = 'YuanKQ'
__mtime__ = 'May 14,2018 15:16'
__mail__ = kq_yuan@outlook.com

__description__==

"""
import pickle
import numpy as np
import tensorflow as tf

def load_dataset_split(path, n_labeled, start=0, sample_size=3, valid_test_ratio=50):
    print("start", start)
    decrease_feature_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_labs = []
    for i in range(start, start+sample_size):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase：", len(increase_feature_matrix), len(increase_feature_matrix[0]), len(increase_labs),
          len(increase_labs[0]))
    for i in range(start, start+sample_size):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))
    print("decrease：", len(decrease_feature_matrix), len(decrease_feature_matrix[0]), len(decrease_labs),
          len(decrease_labs[0]))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count / valid_test_ratio)
    train_count = sample_count - valid_count - test_count

    if n_labeled >= train_count: raise ("The value of n_labeled is lager than the size of train_dataset(%d)." % train_count)
    n_labeled_perclass = int(n_labeled/len(increase_labs))
    x_label = decrease_feature_matrix[0: n_labeled]
    x_label.extend(increase_feature_matrix[0: n_labeled])
    x_unlabel = decrease_feature_matrix[n_labeled: train_count]
    x_unlabel.extend(increase_feature_matrix[n_labeled: train_count])
    y_label = decrease_labs[0: n_labeled]
    y_label.extend(increase_labs[0: n_labeled])
    y_unlabel = decrease_labs[n_labeled: train_count]
    y_unlabel.extend(increase_labs[n_labeled:train_count])

    # ==============================
    valid_x = decrease_feature_matrix[train_count:train_count+valid_count]
    valid_x.extend(increase_feature_matrix[train_count:train_count+valid_count])
    valid_y = decrease_labs[train_count:train_count+valid_count]
    valid_y.extend(increase_labs[train_count:train_count+valid_count])

    test_x = decrease_feature_matrix[train_count+valid_count:]
    test_x.extend(increase_feature_matrix[train_count+valid_count:])
    test_y = decrease_labs[train_count+valid_count:]
    test_y.extend(increase_labs[train_count+valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_label), np.array(y_label), np.array(x_unlabel), np.array(y_unlabel), np.array(valid_x), \
           np.array(valid_y), np.array(test_x), np.array(test_y)

def load_dataset(path, start=0,sample_size=3, valid_test_ratio=50):
    decrease_feature_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_labs = []
    for i in range(start, start + sample_size):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i), 'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase：", len(increase_feature_matrix), len(increase_feature_matrix[0]), len(increase_labs),
          len(increase_labs[0]))
    for i in range(start, start + sample_size):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i), 'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))
    print("decrease：", len(decrease_feature_matrix), len(decrease_feature_matrix[0]), len(decrease_labs),
          len(decrease_labs[0]))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count/valid_test_ratio)
    train_count = sample_count - valid_count - test_count

    x_train = decrease_feature_matrix[0: train_count]
    x_train.extend(increase_feature_matrix[0: train_count])
    y_train = decrease_labs[0: train_count]
    y_train.extend(increase_labs[0:train_count])

    # ==============================
    valid_x = decrease_feature_matrix[train_count:train_count+valid_count]
    valid_x.extend(increase_feature_matrix[train_count:train_count+valid_count])
    valid_y = decrease_labs[train_count:train_count+valid_count]
    valid_y.extend(increase_labs[train_count:train_count+valid_count])

    test_x = decrease_feature_matrix[train_count+valid_count:]
    test_x.extend(increase_feature_matrix[train_count+valid_count:])
    test_y = decrease_labs[train_count+valid_count:]
    test_y.extend(increase_labs[train_count+valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_train), np.array(y_train), np.array(valid_x), \
           np.array(valid_y), np.array(test_x), np.array(test_y)

# if __name__ == '__main__':
#     x_label, y_label, x_unlabel, y_unlabel, valid_x, valid_y, test_x, test_y = load_dataset("/home/yuan/Code/PycharmProjects/vae/ddi/train_dataset", 5000)
#     print("end")

def permute_dataset(arrays):
    """Permute multiple numpy arrays with the same order."""
    if any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError('All arrays must be the same length.')
    random_state = np.random
    order = random_state.permutation(len(arrays[0]))
    return [a[order] for a in arrays]

def reshape_to_4tensor(arrays):
    print("reshape_to_4tensor:", arrays.shape)
    size = arrays.shape[0]
    dim = arrays.shape[1]
    return tf.reshape(arrays, [None, dim, 1, 1])


if __name__ == '__main__':
    load_dataset("/home/yuan/Code/PycharmProjects/vae/ddi/train_dataset")