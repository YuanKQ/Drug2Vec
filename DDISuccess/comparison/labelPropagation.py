# encoding: utf-8

"""
@Author: Kaiqi Yuan
@Since: 2018-08-12 下午8:50
@Description:
"""

import sys
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/comparison'])
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics.pairwise import cosine_similarity
from comparison.dataset_process import load_drugs_list
import random
import tensorflow as tf

def calculate_sim(head, tail, features_dict):
    dim_names = ["actionCode", "atc", "MACCS", "SIDER", "phyCode", "target", "word2vec", "deepwalk"]
    results = []
    for name in dim_names:
        head_vector = [features_dict[head][name]]
        tail_vector = [features_dict[tail][name]]
        cosine = cosine_similarity(head_vector, tail_vector)[0][0]
        results.append(cosine)
    return results

def build_lp_dataset(relations, features_dict):
    idx = list(range(len(relations)))
    # 保证每一次shuffle都能够得到相同的随机数列
    random.seed(10)
    random.shuffle(idx)
    feature_matrix = []
    lab_matrix = []
    for i in idx:
        head = relations[i][0]
        tail = relations[i][1]
        rel = relations[i][2]
        feature_matrix.append(calculate_sim(head, tail, features_dict))
        feature_matrix.append(calculate_sim(tail, head, features_dict))
        if rel == "increase":
            lab = 0
        else:
            lab = 1
        lab_matrix.append(lab)
        lab_matrix.append(lab)
    print("feature_matrix: ", len(feature_matrix), len(feature_matrix[0]))
    print("lab_matrix: ", len(lab_matrix))
    return feature_matrix[15000:30000], lab_matrix[15000:30000]


def get_classfier_result(y_true, y_pred, y_logit):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)

    y_true_matrix = []
    for label in y_true:
        if label == 0:
            y_true_matrix.append([1, 0])
        else:
            y_true_matrix.append([0, 1])
    y_true_matrix = np.array(y_true_matrix)
    print(y_logit)
    auroc_tensor = tf.metrics.auc(y_logit, y_true_matrix)
    aupr_tensor  = tf.metrics.auc(y_logit, y_true_matrix, curve="PR")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        auroc, aupr = sess.run([auroc_tensor, aupr_tensor])
    print("accuracy:{}\nprecision:{}\nrecall:{}\nfscore:{}\nauroc:{}\naupr:{}".format(
        accuracy, precision, recall, fscore, auroc, aupr))
    print("------------------------")


def label_propagation(train_x, train_y, test_x, test_y):
    print("train_x: ", train_x.shape, train_y.shape)
    print("test_x: ", test_x.shape, test_y.shape)
    print("label propagation: ")
    clf = LabelPropagation()
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    y_logit = clf.predict_proba(test_x)
    get_classfier_result(test_y, y_pred, y_logit)


def load_lp_dataset(increase_matrix, increase_labs, decrease_matrix, decrease_labs):
    print("increase： [feature_size] ", len(increase_matrix), len(increase_matrix[0]), "[lab_size] ",
          len(increase_labs))
    print("decrease： [feature_size] ", len(decrease_matrix), len(decrease_matrix[0]), "[lab_size] ",
          len(decrease_labs))

    valid_test_ratio = 50
    n_labeled = 28800//2

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count / valid_test_ratio)
    train_count = sample_count - valid_count - test_count

    x_label = decrease_matrix[0: n_labeled]
    x_label.extend(increase_matrix[0: n_labeled])
    y_label = decrease_labs[0: n_labeled]
    y_label.extend(increase_labs[0: n_labeled])

    test_x = decrease_matrix[train_count + valid_count:]
    test_x.extend(increase_matrix[train_count + valid_count:])
    test_y = decrease_labs[train_count + valid_count:]
    test_y.extend(increase_labs[train_count + valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_label), np.array(y_label), \
           np.array(test_x), np.array(test_y)


if __name__ == '__main__':
    with open("../Data/ddi_rel_v5.pickle", "rb") as rf:
        ddi_increase = pickle.load(rf)
        ddi_decrease = pickle.load(rf)
    # print("increase:", ddi_increase[0])
    # print("decrease:", ddi_decrease[0])
    with open("../Data/drug_features_dict_v5.pickle", "rb") as rf:
        features_dict = pickle.load(rf)
    # print("features.keys(): ", features_dict.keys())
    drugs = load_drugs_list("../Data/drugs_ddi_v5.pickle")

    increase_feature_matrix, increase_labs = build_lp_dataset(ddi_increase, features_dict)
    decrease_feature_matrix, decrease_labs = build_lp_dataset(ddi_decrease, features_dict)
    train_x, train_y, test_x, test_y = load_lp_dataset(increase_feature_matrix, increase_labs, decrease_feature_matrix, decrease_labs)
    label_propagation(train_x, train_y, test_x, test_y)

