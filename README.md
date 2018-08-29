# Drug2Vec
The model of Drug2vec


## DDISuccess/Data
resultant representation of drug properties: drug_#{drug_property}_matrix_dict.pickle

Here is an expamle to load data:
```python    
with open("DDISuccess/Data/drug_SIDER_matrix_dict.pickle", 'rb') as rf:
     drug_SIDER_dict = pickle.load(rf) # key is drug name, value is the representation of side effect
drug_names = list(drug_SIDER_dict.keys())
print(drug_SIDER_dict[drug_names[0]) # get value
```

## Drug2Vec/DDISuccess/BaseHierarchyContext/all_feature_classifier.py
Our model for DDI classifier

## vae/M2
Vae model for DDI classifier.
First run ddi_train_vae.py, then run ddi_train_classifier.py

**[Ref]**
1. [Semi-Supervised Learning with Deep Generative Models](http://arxiv.org/abs/1406.5298)
2. [Implementation by Authors](https://github.com/dpkingma/nips14-ssl)
