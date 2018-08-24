# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: trans_dataset.py
@time: 18-6-21 上午10:28
@description: 对Trans系列的entityEmbeding进行处理。
构建Trans系列的训练集：entity2id.txt, train.txt
从训练结果中提取对应的drug entity embeding:
    entity2vec.bern为TransE训练结果
    entity2vec.txt为PTransE训练结果
"""
import pickle
import numpy as np

def build_dataset():
    # entity2id.txt
    with open("TransDataset/hierarchyEmbeding_id_drugName.pickle", "rb") as rf:
        drugNchem_id_dict = pickle.load(rf)
    
    drug_list = []
    with open("TransDataset/entity2id.txt", "w") as wf:
        index = 0
        for key in drugNchem_id_dict.keys():
            wf.write(key + " " + str(index) + "\n")
            index += 1
            drug_list.append(drugNchem_id_dict[key])
    with open("TransDataset/drugslist_trans_v5.pickle", "wb") as wf:
        pickle.dump(drug_list, wf)
    
    
    #train.txt
    with open("TransDataset/hierarchy.edgelist", "r") as rf:
        lines = rf.readlines()
    with open("TransDataset/train.txt", "w") as wf:
        for line in lines:
            items = line.split()
            wf.write(items[0] + " " +items[1] + " is_a" + "\n")
            

def extract_embeding(target="TransDataset/drug_trans_matrix_dict_v5.pickle", embeding_file="TransDataset/entity2vec.bern"):
    drug_trans_matrix_dict = {}
    with open("TransDataset/drugslist_trans_v5.pickle", "rb") as rf:
        drug_list = pickle.load(rf)
    with open(embeding_file, "r") as rf:
        lines = rf.readlines()
    index = 0
    for line in lines:
        items = line.split()
        values = []
        print(len(items))
        for i in range(127):
            values.append(float(items[i]))
        # drug_trans_matrix_dict[drug_list[index]] = []
        if index < len(drug_list):
            drug_trans_matrix_dict[drug_list[index]] = np.array(values)
        print(index)
        index += 1
    with open(target, "wb") as wf:
        pickle.dump(drug_trans_matrix_dict, wf)


if __name__ == '__main__':
    extract_embeding("TransDataset/drug_PTransE_matrix_dict_v5.pickle", "TransDataset/entity2vec.txt1")
