# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: dataset_process.py
@time: 18-6-22 下午10:02
@description:
"""
import pickle
import random

from utils import shuffle_feature_data
import numpy as np


def build_description_dataset():
    with open("../Data/drug_description_word2ve_v5.pickle", "rb") as rf:
        drug_description = pickle.load(rf)

    with open("../Data/ddi_rel_v5.pickle", "rb") as rf:
        ddi_increase = pickle.load(rf)
        ddi_decrease = pickle.load(rf)

    def shuffle_data(relations, drug_all_dict):
        idx = list(range(len(relations)))
        random.seed(10)
        random.shuffle(idx)
        head_feature_matrix = []
        tail_feature_matrix = []
        for i in idx:
            head = relations[i][0]
            tail = relations[i][1]
            head_feature_matrix.append(drug_all_dict[head])
            head_feature_matrix.append(drug_all_dict[tail])
            tail_feature_matrix.append(drug_all_dict[head])
            tail_feature_matrix.append(drug_all_dict[tail])
        return head_feature_matrix, tail_feature_matrix

    increase_head_description_matrix, increase_tail_description_matrix = shuffle_data(ddi_increase, drug_description)
    decrease_head_description_matrix, decrease_tail_description_matrix = shuffle_data(ddi_decrease, drug_description)

    def dump_dataset(head_feature_matrix, tail_feature_matrix, target_file_prefix):
        partition = 5000
        print(target_file_prefix)
        for i in range(3, 6):
            start = i * partition
            end = (i + 1) * partition
            with open("%s_%d.pickle" % (target_file_prefix, i), "wb") as wf:
                pickle.dump(head_feature_matrix[start:end], wf)
                pickle.dump(tail_feature_matrix[start:end], wf)
            print("start: %d, end: %d" % (start, end))

    dump_dataset(increase_head_description_matrix, increase_tail_description_matrix, "DescriptionDataset/increase_description_matrix")
    dump_dataset(decrease_head_description_matrix, decrease_tail_description_matrix, "DescriptionDataset/decrease_description_matrix")


def load_context(path, n_labeled):
    valid_test_ratio = 50
    head_decrease_feature_matrix = []
    tail_decrease_feature_matrix = []
    head_increase_feature_matrix = []
    tail_increase_feature_matrix = []
    for i in range(3, 6):
        with open(
                "%s/increase_description_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            head_increase_feature_matrix.extend(pickle.load(rf))
            tail_increase_feature_matrix.extend(pickle.load(rf))
    print("increase：", len(head_increase_feature_matrix))
    for i in range(3, 6):
        with open(
                "%s/decrease_description_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            head_decrease_feature_matrix.extend(pickle.load(rf))
            tail_decrease_feature_matrix.extend(pickle.load(rf))
    print("decrease：", len(head_decrease_feature_matrix))

    sample_count = len(head_increase_feature_matrix)
    print(sample_count)
    valid_count = test_count = int(sample_count / valid_test_ratio)
    train_count = sample_count - valid_count - test_count
    # print("traincount: ", train_count)

    x_label_head = head_decrease_feature_matrix[0: n_labeled]
    x_label_head.extend(head_increase_feature_matrix[0: n_labeled])
    x_label_tail = tail_decrease_feature_matrix[0: n_labeled]
    x_label_tail.extend(tail_increase_feature_matrix[0: n_labeled])

    # valid_x = decrease_feature_matrix[train_count:train_count + valid_count]
    # valid_x.extend(increase_feature_matrix[train_count:train_count + valid_count])
    # valid_y = decrease_labs[train_count:train_count + valid_count]
    # valid_y.extend(increase_labs[train_count:train_count + valid_count])

    test_x_head = head_decrease_feature_matrix[train_count + valid_count:]
    test_x_head.extend(head_increase_feature_matrix[train_count + valid_count:])
    test_x_tail = tail_decrease_feature_matrix[train_count + valid_count:]
    test_x_tail.extend(tail_increase_feature_matrix[train_count + valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_label_head), np.array(x_label_tail), np.array(test_x_head), np.array(test_x_tail)


if __name__ == '__main__':
    # build_description_dataset()
    n_labeled = 1000
    train_head_context, train_tail_context, test_head_context, test_tail_context = load_context("DescriptionDataset", n_labeled//2)
    print(train_head_context.shape)
    print(train_tail_context.shape)
    print(test_head_context.shape)
    print(test_tail_context.shape)
    print("end")