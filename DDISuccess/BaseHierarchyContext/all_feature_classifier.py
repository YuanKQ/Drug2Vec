# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: all_feature_classifier.py
@time: 18-6-22 下午10:40
@description:
"""

import sys
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/BaseHierarchyContext'])
import time
from BaseHierarchyContext.dataset_process import load_context
from BaseHierarchy.dataset_process import load_dataset
from utils import permute_dataset, fully_connected_layer, convolutional_layer, lenet5, BiRNN, update_best_results, \
    calculate_f1, next_batch
import tensorflow as tf
import prettytensor as pt

batch_size = 40
epochs = 600  # Number of epochs through the full dataset
learning_rate = 3e-3  # 原来的学习率为3e-4
num_lab = 1440  #1440 4320 7200  14400  21600 24800 28800
dim_y = 2
latent_dim = 500
classify_dataset_path = "/data/cdy/ykq/BaseHierarchyDataset"
context_datset_path = "/data/cdy/ykq/DescriptionDataset"
hierarchy_types = 10 # deepwalk, node2vec, LINE, TransE, PTransE
# clatent_dim = 100  # num_classes
num_hidden = 100  # hidden layer num of features from BiLSTM

train_data, train_hierarchy, train_labels, test_data, test_hierarchy, test_labels = load_dataset(classify_dataset_path, num_lab//2)
train_head_context, train_tail_context, test_head_context, test_tail_context = load_context(context_datset_path, num_lab//2)
word2vec_dim = train_head_context.shape[-1]  # num_input
sentence_len = train_head_context.shape[1]  # time_steps
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
print("train_head_context: ", train_head_context.shape, "train_tail_context: ", train_tail_context.shape,
      "test_head_context: ", test_head_context.shape, "test_tail_context: ", test_tail_context.shape)

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

# tf Graph input
head_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
head_hierarchy_placeholder = tf.placeholder(tf.float32, [batch_size, hierarchy_dim])
head_context_placeholder = tf.placeholder(tf.float32, [batch_size, sentence_len, word2vec_dim])
tail_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
tail_hierarchy_placeholder = tf.placeholder(tf.float32, [batch_size, hierarchy_dim])
tail_context_placeholder = tf.placeholder(tf.float32, [batch_size, sentence_len, word2vec_dim])
labels_placeholder    = tf.placeholder(tf.float32, [batch_size, dim_y])


head_trans = convolutional_layer(head_data_placeholder, latent_dim)
tail_trans = convolutional_layer(tail_data_placeholder, latent_dim)
print("head_trans: ", head_trans.shape)
print("tail_trans: ", tail_trans.shape)

head_context_trans = BiRNN(head_context_placeholder, sentence_len, word2vec_dim, num_hidden)
tail_context_trans = BiRNN(tail_context_placeholder, sentence_len, word2vec_dim, num_hidden)
clatent_dim = head_context_trans.shape[-1]
print("clatent_dim: ", clatent_dim)

input_data = tf.reshape(tf.concat([head_trans, head_hierarchy_placeholder, head_context_trans,
                                   tail_trans, tail_hierarchy_placeholder, tail_context_trans], 1),
                        [batch_size, latent_dim*2+hierarchy_dim*2+clatent_dim*2, 1, 1])
# input_data = tf.reshape(tf.concat([head_trans, head_context_trans,
#                                    tail_trans, tail_context_trans], 1),
#                         [batch_size, latent_dim*2+clatent_dim*2, 1, 1])

print("input_data shape: ", input_data.shape)


result = lenet5(input_data, labels_placeholder, dim_y)
# print("@@@@@result:", result.softmax.shape)

pred_logit = result.softmax.softmax_activation()
accuracy_tensor = result.softmax.evaluate_classifier(labels_placeholder, phase=pt.Phase.test)
precision_tensor, recall_tensor = result.softmax.evaluate_precision_recall(labels_placeholder, phase=pt.Phase.test)
_, auroc_tensor = tf.metrics.auc(labels_placeholder, pred_logit)
_, aupr_tensor  = tf.metrics.auc(labels_placeholder, pred_logit, curve="PR")


optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

runner = pt.train.Runner()
best_f1 = 0
best_accuracy = 0
best_f1_results = {"precision": 0, "recall": 0, "auroc": 0, "aupr": 0, "accuracy": 0, "epoch": 0, "f1":0}
best_accuracy_results = {"precision": 0, "recall": 0, "auroc": 0, "aupr": 0, "accuracy": 0, "epoch": 0, "f1":0}
start = time.time()
train_head_hierarchy = train_head_deepwalk
train_tail_hierarchy = train_tail_deepwalk
test_head_hierarchy  = test_head_deepwalk
test_tail_hierarchy  = test_tail_deepwalk
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        train_head_data, train_head_hierarchy, train_head_context,\
        train_tail_data, train_tail_hierarchy, train_tail_context, \
        train_labels = permute_dataset((train_head_data, train_head_hierarchy, train_head_context,
                                        train_tail_data, train_tail_hierarchy, train_tail_context,
                                        train_labels))

        runner.train_model(train_op, result.loss, train_batch_num,
                           feed_vars=(head_data_placeholder, head_hierarchy_placeholder, head_context_placeholder,
                                      tail_data_placeholder, tail_hierarchy_placeholder, tail_context_placeholder,
                                      labels_placeholder),
                           feed_data=pt.train.feed_numpy(batch_size,
                                                         train_head_data, train_head_hierarchy, train_head_context,
                                                         train_tail_data, train_tail_hierarchy, train_tail_context,
                                                         train_labels),
                           print_every=500)
        accuracy = 0
        precision = 0
        recall = 0
        auroc = 0
        aupr  = 0
        num_batches = test_labels.shape[0]/batch_size
        for test_batch_head_data, test_batch_head_hierarchy, test_batch_head_context, \
            test_batch_tail_data, test_batch_tail_hierarchy, test_batch_tail_context, \
            test_batch_labels in next_batch(test_head_data, test_head_hierarchy, test_head_context, \
                                            test_tail_data, test_tail_hierarchy, test_tail_context, \
                                            test_labels, batch_size):
            accuracy += accuracy_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                       head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                       head_context_placeholder:   test_batch_head_context,
                                                       tail_data_placeholder:      test_batch_tail_data,
                                                       tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                       tail_context_placeholder:   test_batch_tail_context,
                                                       labels_placeholder:         test_batch_labels})
            precision += precision_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                         head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                         head_context_placeholder:   test_batch_head_context,
                                                         tail_data_placeholder:      test_batch_tail_data,
                                                         tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                         tail_context_placeholder:   test_batch_tail_context,
                                                         labels_placeholder:         test_batch_labels}) / 2
            recall += recall_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                   head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                   head_context_placeholder:   test_batch_head_context,
                                                   tail_data_placeholder:      test_batch_tail_data,
                                                   tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                   tail_context_placeholder:   test_batch_tail_context,
                                                   labels_placeholder:         test_batch_labels}) / 2
            auroc += auroc_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                 head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                 head_context_placeholder:   test_batch_head_context,
                                                 tail_data_placeholder:      test_batch_tail_data,
                                                 tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                 tail_context_placeholder:   test_batch_tail_context,
                                                 labels_placeholder:         test_batch_labels})
            aupr += aupr_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                               head_hierarchy_placeholder: test_batch_head_hierarchy,
                                               head_context_placeholder:   test_batch_head_context,
                                               tail_data_placeholder:      test_batch_tail_data,
                                               tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                               tail_context_placeholder:   test_batch_tail_context,
                                               labels_placeholder:         test_batch_labels})
        # classification_accuracy = runner.evaluate_model(accuracy_tensor, test_batch_num,
        #                                                 feed_vars=(head_data_placeholder, head_hierarchy_placeholder, head_context_placeholder,
        #                                                            tail_data_placeholder, tail_hierarchy_placeholder, tail_context_placeholder,
        #                                                            labels_placeholder),
        #                                                 feed_data=pt.train.feed_numpy(batch_size,
        #                                                                               test_head_data, test_head_hierarchy, test_head_context,
        #                                                                               test_tail_data, test_tail_hierarchy, test_tail_context,
        #                                                                               test_labels),
        #                                                 print_every=500)

        accuracy /= num_batches
        precision /= num_batches
        recall /= num_batches
        auroc /= num_batches
        aupr /= num_batches
        f1 = calculate_f1(precision, recall)
        if best_f1 < f1:
            best_f1 = f1
            update_best_results(accuracy, precision, recall, f1, aupr, auroc, epoch, best_f1_results)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            update_best_results(accuracy, precision, recall, f1, aupr, auroc, epoch, best_accuracy_results)
        print('Accuracy after %d epoch %g%%' % (epoch + 1, accuracy * 100))
        print('F1 after %d epoch %g%%' % (epoch + 1, f1 * 100))
    print('Train size is {}.Best accuracy is: {}'.format(num_lab, best_accuracy_results))
    print('Train size is {}.Best f1 is: {}'.format(num_lab, best_f1_results))
    print('==================================')
end = time.time()
print("elapse: ", (end - start) / 3600)