# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: vec_transfer_classifier.py
@time: 18-6-13 下午9:25
@description: 药物特征（flatten为一维）经过处理后再拼接在一起输入到conv分类器中
"""
import sys
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/Base'])
import time
from BaseHierarchy.dataset_process import load_dataset
from utils import permute_dataset, fully_connected_layer, convolutional_layer, lenet5
import tensorflow as tf
import prettytensor as pt

batch_size = 40
epochs = 800  # Number of epochs through the full dataset
learning_rate = 3e-3  # 原来的学习率为3e-4
num_lab = 24800
dim_y = 2
latent_dim = 500
classify_dataset_path = "/data/cdy/ykq/BaseHierarchyDataset"
hierarchy_types = 10 # deepwalk, node2vec, LINE, TransE, PTransE

train_data, train_hierarchy, train_labels, test_data, test_hierarchy, test_labels = load_dataset(classify_dataset_path, num_lab//2)
hierarchy_dim = train_hierarchy.shape[1] // hierarchy_types
dim_x = train_data.shape[1]
feature_dim = int(dim_x // 2)
train_batch_num = train_data.shape[0] // batch_size
test_batch_num = test_data.shape[0] // batch_size
train_head_data = train_data[:, 0:feature_dim]
train_tail_data = train_data[:, feature_dim:dim_x]
test_head_data = test_data[:, 0:feature_dim]
test_tail_data = test_data[:, feature_dim:dim_x]
print("dim_x: ", dim_x, "\ntrain_size:", train_data.shape[0], "\ntest_size:",test_data.shape[0])

# extract_hierarchy:
def load_hierarchy(hierarchy_matrix):
    head_deepwalk = hierarchy_matrix[:, 0: hierarchy_dim]
    head_node2vec = hierarchy_matrix[:, hierarchy_dim: hierarchy_dim * 2]
    head_LINE     = hierarchy_matrix[:, hierarchy_dim * 2: hierarchy_dim * 3]
    head_Trans    = hierarchy_matrix[:, hierarchy_dim * 3: hierarchy_dim * 4]
    head_PTransE  = hierarchy_matrix[:, hierarchy_dim * 4: hierarchy_dim * 5]
    tail_deepwalk = hierarchy_matrix[:, hierarchy_dim * 5: hierarchy_dim * 6]
    tail_node2vec = hierarchy_matrix[:, hierarchy_dim * 6: hierarchy_dim * 7]
    tail_LINE     = hierarchy_matrix[:, hierarchy_dim * 7: hierarchy_dim * 8]
    tail_Trans    = hierarchy_matrix[:, hierarchy_dim * 8: hierarchy_dim * 9]
    tail_PTransE  = hierarchy_matrix[:, hierarchy_dim * 9: hierarchy_dim * 10]
    return head_deepwalk, head_node2vec, head_LINE, head_Trans, head_PTransE, \
           tail_deepwalk, tail_node2vec, tail_LINE, tail_Trans, tail_PTransE

train_head_deepwalk, train_head_node2vec, train_head_LINE, train_head_Trans, train_head_PTransE, \
train_tail_deepwalk, train_tail_node2vec, train_tail_LINE, train_tail_Trans, train_tail_PTransE = load_hierarchy(train_hierarchy)
test_head_deepwalk, test_head_node2vec, test_head_LINE, test_head_Trans, test_head_PTransE, \
test_tail_deepwalk, test_tail_node2vec, test_tail_LINE, test_tail_Trans, test_tail_PTransE= load_hierarchy(test_hierarchy)

head_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
head_hierarchy_placeholder = tf.placeholder(tf.float32, [batch_size, hierarchy_dim])
tail_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
tail_hierarchy_placeholder = tf.placeholder(tf.float32, [batch_size, hierarchy_dim])
labels_placeholder    = tf.placeholder(tf.float32, [batch_size, dim_y])

head_trans = convolutional_layer(head_data_placeholder, latent_dim)
tail_trans = convolutional_layer(tail_data_placeholder, latent_dim)
print("head_trans: ", head_trans.shape)
print("tail_trans: ", tail_trans.shape)

input_data = tf.reshape(tf.concat([head_trans, head_hierarchy_placeholder, tail_trans, tail_hierarchy_placeholder], 1),
                        [batch_size, latent_dim*2+hierarchy_dim*2, 1, 1])
print("input_data shape: ", input_data.shape)


result = lenet5(input_data, labels_placeholder, dim_y)
accuracy = result.softmax.evaluate_classifier(labels_placeholder, phase=pt.Phase.test)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

runner = pt.train.Runner()
best_accuracy = 0
best_epoch = 0
start = time.time()
train_head_hierarchy = train_head_deepwalk
train_tail_hierarchy = train_tail_deepwalk
test_head_hierarchy  = test_head_deepwalk
test_tail_hierarchy  = test_tail_deepwalk
with tf.Session() as sess:
    for epoch in range(epochs):
        train_head_data, train_head_hierarchy, \
        train_tail_data, train_tail_hierarchy, \
        train_labels = permute_dataset((train_head_data, train_head_hierarchy,
                                        train_tail_data, train_tail_hierarchy,
                                        train_labels))

        runner.train_model(train_op, result.loss, train_batch_num,
                           feed_vars=(head_data_placeholder, head_hierarchy_placeholder,
                                      tail_data_placeholder, tail_hierarchy_placeholder, labels_placeholder),
                           feed_data=pt.train.feed_numpy(batch_size,
                                                         train_head_data, train_head_hierarchy,
                                                         train_tail_data, train_tail_hierarchy, train_labels),
                           print_every=500)

        classification_accuracy = runner.evaluate_model(accuracy, test_batch_num,
                                                        feed_vars=(head_data_placeholder, head_hierarchy_placeholder,
                                                                   tail_data_placeholder, tail_hierarchy_placeholder, labels_placeholder),
                                                        feed_data=pt.train.feed_numpy(batch_size,
                                                                                      test_head_data, test_head_hierarchy,
                                                                                      test_tail_data, test_tail_hierarchy, test_labels),
                                                        print_every=500)

        if best_accuracy < classification_accuracy[0]:
            best_accuracy = classification_accuracy
            best_epoch = epoch
        print('Accuracy after %d epoch %g%%' % (epoch + 1, classification_accuracy[0] * 100))

    print('Train size is {}.Best accuracy is {}%% at {} epoch.'.format(num_lab, best_accuracy, best_epoch))
    print('==================================')
end = time.time()
print("elapse: ", (end-start)/3600)

