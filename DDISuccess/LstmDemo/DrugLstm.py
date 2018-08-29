# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: DrugLstm.py
@time: 18-7-15 上午9:19
@description:
"""

from BaseHierarchyContext.dataset_process import load_context

context_datset_path = "/data/cdy/ykq/DescriptionDataset"
num_lab = 24800
train_head_context, train_tail_context, test_head_context, test_tail_context = load_context(context_datset_path, num_lab//2)

word2vec_dim = train_head_context.shape[-1]  # num_input
sentence_len = train_head_context.shape[1]  # time_steps