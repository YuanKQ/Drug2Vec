# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: add_context_info.py
@time: 18-6-22 下午9:41
@description:
"""
from BaseHierarchyContext.dataset_process import load_dataset
import tensorflow as tf

from utils import BiRNN

num_lab = 24800
batch_size = 100
context_path = ""
train_data, train_hierarchy, train_context, train_labels, test_data, test_hierarchy, test_context, test_labels = load_dataset(context_path, num_lab//2)

# train_context.shape = [train_size, sentence_len, word2vec_dim]
clatent_dim = 100  # num_classes
num_hidden = 128  # hidden layer num of features from BiLSTM
word2vec_dim = train_context.shape[-1]  # num_input
sentence_len = train_context.shape[1]  # time_steps

# tf Graph input
head_context_placeholder = tf.placeholder(tf.float32, [batch_size, sentence_len, word2vec_dim])
tail_context_placeholder = tf.placeholder(tf.float32, [batch_size, sentence_len, word2vec_dim])

# weights of BiLSTM
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, clatent_dim]))
}
biases = {
    'out': tf.Variable(tf.random_normal([clatent_dim]))
}
head_context_trans = BiRNN(head_context_placeholder, sentence_len, num_hidden, weights, biases)
tail_context_trans = BiRNN(tail_context_placeholder, sentence_len, num_hidden, weights, biases)
