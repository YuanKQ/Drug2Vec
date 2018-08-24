# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: base_dataset_build.py
@time: 18-6-13 上午8:28
@description: 将所有特征拼接在一块儿实验数据集
"""
import sys
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/BaseHierarchyContext'])

import pickle
import numpy as np

from utils import shuffle_feature_data, dump_dataset, load_drugs_list

with open("../Data/ddi_rel_v5.pickle", "rb") as rf:
    ddi_increase = pickle.load(rf)
    ddi_decrease = pickle.load(rf)
# print("increase:", ddi_increase[0])
# print("decrease:", ddi_decrease[0])

with open("../Data/drug_features_dict_v5.pickle", "rb") as rf:
    features_dict = pickle.load(rf)

drugs = load_drugs_list("../Data/drugs_ddi_v5.pickle")
drug_features_dict = {}
for drug in drugs:
    drug_features_dict[drug] = np.concatenate((features_dict[drug]["actionCode"],
                                              features_dict[drug]["atc"],
                                              features_dict[drug]["MACCS"],
                                              features_dict[drug]["SIDER"],
                                              features_dict[drug]["phyCode"],
                                              features_dict[drug]["target"]),
                                              )

increase_feature_matrix, increase_lab_matrix = shuffle_feature_data(ddi_increase, drug_features_dict)
decrease_feature_matrix, decrease_lab_matrix = shuffle_feature_data(ddi_decrease, drug_features_dict)
dump_dataset(increase_feature_matrix, increase_lab_matrix, "BaseDataset/increase_features_labs_matrix")
dump_dataset(decrease_feature_matrix, decrease_lab_matrix, "BaseDataset/decrease_features_labs_matrix")

## TEST
# with open("BaseDataset/decrease_features_labs_matrix_0.pickle", "rb") as rf:
#     drug1 = pickle.load(rf)
# with open("BaseDataset/decrease_features_labs_matrix_1.pickle", "rb") as rf:
#     drug2 = pickle.load(rf)
