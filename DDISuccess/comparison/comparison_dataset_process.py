# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: base_dataset_build.py
@time: 18-6-13 上午8:28
@description: base实验数据集
"""

import sys
# sys.path.extend(['/home/cdy/ykq/DDISuccess', "/home/cdy/yuan/DDISuccess/comparison"])
# sys.path.extend(['/home/cdy/ykq/DDISuccess', "/home/cdy/yuan/DDISuccess/comparison"])
import pickle
import numpy as np

from utils import shuffle_feature_data, dump_dataset, load_drugs_list, load_dataset_split

def build_comparison_dataset():
    with open("../Data/ddi_rel_v5.pickle", "rb") as rf:
        ddi_increase = pickle.load(rf)
        ddi_decrease = pickle.load(rf)
    # print("increase:", ddi_increase[0])
    # print("decrease:", ddi_decrease[0])

    with open("../Data/drug_features_dict_v5.pickle", "rb") as rf:
        features_dict = pickle.load(rf)

    drugs = load_drugs_list("../Data/drugs_ddi_v5.pickle")
    drug_features_dict = {}
    index = 0
    print(features_dict.keys())
    for drug in drugs:
        drug_features_dict[drug] = np.concatenate((features_dict[drug]["actionCode"],
                                                  features_dict[drug]["atc"],
                                                  features_dict[drug]["MACCS"],
                                                  features_dict[drug]["SIDER"],
                                                  features_dict[drug]["phyCode"],
                                                  features_dict[drug]["target"],
                                                  features_dict[drug]["word2vec"],
                                                  features_dict[drug]["deepwalk"]
                                                  ))
        index += 1
        if index == 1:
            print(len(drug_features_dict[drug]))
        if index == 10:
            break

    increase_feature_matrix, increase_lab_matrix = shuffle_feature_data(ddi_increase, drug_features_dict)
    decrease_feature_matrix, decrease_lab_matrix = shuffle_feature_data(ddi_decrease, drug_features_dict)
    dump_dataset(increase_feature_matrix, increase_lab_matrix, "/home/cdy/ykq/DDISuccess/comparison/ComparisonDataset/increase_features_labs_matrix")
    dump_dataset(decrease_feature_matrix, decrease_lab_matrix, "/home/cdy/ykq/DDISuccess/comparison/ComparisonDataset/decrease_features_labs_matrix")

def load_comparison_dataset(path, n_labeled):
    valid_test_ratio = 50
    decrease_feature_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_labs = []
    start_index = 3
    end_index = 6
    for i in range(start_index, end_index):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase： [feature_size] ", len(increase_feature_matrix),len(increase_feature_matrix[0]), "[lab_size] ", len(increase_labs))
    for i in range(start_index, end_index):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count / valid_test_ratio)
    train_count = sample_count - valid_count - test_count

    x_label = decrease_feature_matrix[0: n_labeled]
    x_label.extend(increase_feature_matrix[0: n_labeled])
    y_label = decrease_labs[0: n_labeled]
    y_label.extend(increase_labs[0: n_labeled])

    test_x = decrease_feature_matrix[train_count + valid_count:]
    test_x.extend(increase_feature_matrix[train_count + valid_count:])
    test_y = decrease_labs[train_count + valid_count:]
    test_y.extend(increase_labs[train_count + valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_label), np.array(y_label), \
           np.array(test_x), np.array(test_y)

## TEST
# with open("BaseDataset/decrease_features_labs_matrix_0.pickle", "rb") as rf:
#     drug1 = pickle.load(rf)
# with open("BaseDataset/decrease_features_labs_matrix_1.pickle", "rb") as rf:
#     drug2 = pickle.load(rf)
if __name__ == '__main__':
    x_label, y_label, test_x, test_y = load_comparison_dataset("ComparisonDataset", 2)
    print("x_label:", len(x_label), len(x_label[0]))
    print("y_label:", len(y_label), len(y_label[0]))
    print("test_x:", len(test_x), len(test_x[0]))
    print("test_y:", len(test_y), len(test_y[0]))