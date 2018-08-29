# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: dataset_process.py
@time: 18-6-20 上午11:35
@description:
"""

import pickle
import numpy as np
import random

# from utils import shuffle_feature_hierarchy_data, load_drugs_list

start_index = 3
end_index = 6


def load_drugs_list(ddi_file):
    with open(ddi_file, 'rb') as rf:
        drugs_list = pickle.load(rf)
    return drugs_list


def shuffle_feature_hierarchy_data(relations, drug_all_dict, network_dict):
    idx = list(range(len(relations)))
    # 保证每一次shuffle都能够得到相同的随机数列
    random.seed(10)
    random.shuffle(idx)
    feature_matrix = []
    lab_matrix = []
    network_matrix = []
    for i in idx:
        head = relations[i][0]
        tail = relations[i][1]
        rel = relations[i][2]
        feature_matrix.append(np.concatenate((drug_all_dict[head], drug_all_dict[tail])))
        network_matrix.append(np.concatenate((network_dict[head], network_dict[tail])))
        feature_matrix.append(np.concatenate((drug_all_dict[tail], drug_all_dict[head])))
        network_matrix.append(np.concatenate((network_dict[tail], network_dict[head])))
        if rel == "increase":
            lab = np.array([0,1])
        else:
            lab = np.array([1, 0])
        lab_matrix.append(lab)
        lab_matrix.append(lab)
    print("feature_matrix: ", len(feature_matrix), len(feature_matrix[0]))
    print("network_matrix: ", len(network_matrix), len(network_matrix[0]))
    print("lab_matrix: ", len(lab_matrix), len(lab_matrix[0]))
    return feature_matrix, network_matrix, lab_matrix

def build_dataset():
    with open("../Data/ddi_rel_v5.pickle", "rb") as rf:
        ddi_increase = pickle.load(rf)
        ddi_decrease = pickle.load(rf)
    # print("increase:", ddi_increase[0])
    # print("decrease:", ddi_decrease[0])

    with open("../Data/drug_features_dict_v5.pickle", "rb") as rf:
        features_dict = pickle.load(rf)
    with open("TransDataset/drug_trans_matrix_dict_v5.pickle", "rb") as rf:
        drug_trans_matrix = pickle.load(rf)

    with open("TransDataset/drug_PTransE_matrix_dict_v5.pickle", "rb") as rf:
        drug_ptranse_matrix = pickle.load(rf)

    drugs = load_drugs_list("../Data/drugs_ddi_v5.pickle")
    drug_features_dict = {}
    drug_network_dict = {}
    for drug in drugs:
        drug_features_dict[drug] = np.concatenate((features_dict[drug]["actionCode"],
                                                  features_dict[drug]["atc"],
                                                  features_dict[drug]["MACCS"],
                                                  features_dict[drug]["SIDER"],
                                                  features_dict[drug]["phyCode"],
                                                  features_dict[drug]["target"],
                                                  # features_dict[drug]["word2vec"]
                                                  )
                                                  )
        drug_network_dict[drug] = np.concatenate((features_dict[drug]["deepwalk"], features_dict[drug]["node2vec"],
                                                  features_dict[drug]["LINE"], drug_trans_matrix[drug],
                                                  drug_ptranse_matrix[drug]))

    increase_feature_matrix, increase_hierarchy_matrix, increase_lab_matrix = shuffle_feature_hierarchy_data(ddi_increase, drug_features_dict, drug_network_dict)
    decrease_feature_matrix, decrease_hierarchy_matrix, decrease_lab_matrix = shuffle_feature_hierarchy_data(ddi_decrease, drug_features_dict, drug_network_dict)

    def dump_dataset(feature_matrix, hierarchy_matrix, lab_matrix, target_file_prefix):
        partition = 5000
        start = 0
        end = 0
        count = len(lab_matrix) // partition
        print(target_file_prefix)
        # for i in range(count):
        for i in range(start_index, end_index):
            start = i * partition
            end = (i + 1) * partition
            with open("%s_%d.pickle" % (target_file_prefix, i), "wb") as wf:
                pickle.dump(feature_matrix[start:end], wf)
                pickle.dump(hierarchy_matrix[start:end], wf)
                pickle.dump(lab_matrix[start:end], wf)
            print("start: %d, end: %d" % (start, end))
            start = end

    dump_dataset(increase_feature_matrix, increase_hierarchy_matrix, increase_lab_matrix, "BaseWordvecHierarchyDataset/increase_features_labs_matrix")
    dump_dataset(decrease_feature_matrix, decrease_hierarchy_matrix, decrease_lab_matrix, "BaseWordvecHierarchyDataset/decrease_features_labs_matrix")


def load_dataset(path, n_labeled):
    valid_test_ratio = 50
    decrease_feature_matrix = []
    decrease_hierarchy_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_hierarchy_matrix = []
    increase_labs = []
    for i in range(start_index, end_index):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_hierarchy_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase：", len(increase_feature_matrix), len(increase_feature_matrix[0]), len(increase_labs),
          len(increase_labs[0]))
    for i in range(start_index, end_index):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_hierarchy_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))
    print("decrease：", len(decrease_feature_matrix), len(decrease_feature_matrix[0]), len(decrease_labs),
          len(decrease_labs[0]))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count / valid_test_ratio)
    train_count = sample_count - valid_count - test_count
    # print("traincount: ", train_count)

    x_label = decrease_feature_matrix[0: n_labeled]
    x_label.extend(increase_feature_matrix[0: n_labeled])
    x_hierarchy_label = decrease_hierarchy_matrix[0: n_labeled]
    x_hierarchy_label.extend(increase_hierarchy_matrix[0: n_labeled])
    y_label = decrease_labs[0: n_labeled]
    y_label.extend(increase_labs[0: n_labeled])

    # valid_x = decrease_feature_matrix[train_count:train_count + valid_count]
    # valid_x.extend(increase_feature_matrix[train_count:train_count + valid_count])
    # valid_y = decrease_labs[train_count:train_count + valid_count]
    # valid_y.extend(increase_labs[train_count:train_count + valid_count])

    test_x = decrease_feature_matrix[train_count + valid_count:]
    test_x.extend(increase_feature_matrix[train_count + valid_count:])
    test_x_hierarchy = decrease_hierarchy_matrix[train_count + valid_count:]
    test_x_hierarchy.extend(increase_hierarchy_matrix[train_count+valid_count:])
    test_y = decrease_labs[train_count + valid_count:]
    test_y.extend(increase_labs[train_count + valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_label), np.array(x_hierarchy_label), np.array(y_label), \
           np.array(test_x), np.array(test_x_hierarchy), np.array(test_y)



if __name__ == '__main__':
    build_dataset()
    x_label, x_h_label, y_label, test_x, test_x_h, test_y = load_dataset("BaseWordvecHierarchyDataset", 1000)
    print("x_label:", len(x_label), len(x_label[0]))
    print("x_h_label:", len(x_h_label), len(x_h_label[0]))
    print("y_label:", len(y_label), len(y_label[0]))
    print("test_x:", len(test_x), len(test_x[0]))
    print("test_x_h:", len(test_x_h), len(test_x_h[0]))
    print("test_y:", len(test_y), len(test_y[0]))