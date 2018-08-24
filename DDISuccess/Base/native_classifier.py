# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: native_classifier.py
@time: 18-6-13 下午6:25
@description: 直接将药物特征扔到conv分类器中
"""
import sys
import time

sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/Base'])

import numpy as np

from utils import load_dataset_split, permute_dataset, lenet5, calculate_f1, update_best_results
import tensorflow as tf
import prettytensor as pt

#num_batches = 100  # Number of minibatches in a single epoch
batch_size = 40
epochs = 10# 800  # Number of epochs through the full dataset
learning_rate = 3e-4
num_lab = 14400 #1440  4320 7200  14400  21600 24800
classify_dataset_path = "/data/cdy/ykq/BaseDataset/TestDataset"
# classify_dataset_path = "/home/yuan/Code/PycharmProjects/vae/ddi/test_dataset"
# train_data, train_labels, _, __, test_data, test_labels = load_dataset(classify_dataset_path, 3)
train_data, train_labels, _, __, test_data, test_labels = load_dataset_split(classify_dataset_path, num_lab//2, 3)
dim_x = train_data.shape[1]
# one_dim = dim_x / 2
# head_feature = train_data[:, 0:one_dim]
# tail_feature = train_data[:, one_dim: dim_x]
# head_trans_placeholder = tf.placeholder(tf.float32, [])
dim_y = 2
train_size = train_data.shape[0]
test_size = test_data.shape[0]
print("dim_x: ", dim_x, "\ntrain_size:", train_size, "\ntest_size:",test_size)

train_data = np.reshape(train_data, [train_size, dim_x, 1, 1])
test_data = np.reshape(test_data, [test_size, dim_x, 1, 1])
assert train_size % batch_size == 0, '#TrainSize % #BatchSize != 0'
assert test_size % batch_size == 0, '#TestSize % #BatchSize != 0'
train_batch_num = int(train_size / batch_size)
test_batch_num = int(test_size / batch_size)

data_placeholder = tf.placeholder(tf.float32, [None, dim_x, 1, 1])
labels_placeholder = tf.placeholder(tf.float32, [None, dim_y])

result = lenet5(data_placeholder, labels_placeholder, dim_y)
pred_logit = result.softmax.softmax_activation()
accuracy_tensor = result.softmax.evaluate_classifier(labels_placeholder, phase=pt.Phase.test)
precision_tensor, recall_tensor = result.softmax.evaluate_precision_recall(labels_placeholder, phase=pt.Phase.test)
_, auroc_tensor = tf.metrics.auc(labels_placeholder, pred_logit)
_, aupr_tensor  = tf.metrics.auc(labels_placeholder, pred_logit, curve="PR")

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

#save_path = '/data/cdy/ykq/checkpoints/model_conv2d_{}-{}.cpkt'.format(
#            learning_rate, time.strftime("%m-%d-%H%M%S", time.localtime()))
#print("model has been saved: " + save_path)
#runner = pt.train.Runner(save_path)
runner = pt.train.Runner()
best_f1 = 0
best_accuracy = 0
best_f1_results = {"precision": 0, "recall": 0, "auroc": 0, "aupr": 0, "accuracy": 0, "epoch": 0, "f1":0}
best_accuracy_results = {"precision": 0, "recall": 0, "auroc": 0, "aupr": 0, "accuracy": 0, "epoch": 0, "f1":0}
start = time.time()
with tf.Session() as sess:
    # print(epochs)
    for epoch in range(epochs):
        train_data, train_labels = permute_dataset((train_data, train_labels))

        # 并没有保存最佳的model
        runner.train_model(train_op, result.loss, batch_size,
                           feed_vars=(data_placeholder, labels_placeholder),
                           feed_data=pt.train.feed_numpy(train_batch_num, train_data, train_labels),
                           print_every=500)
        accuracy  = accuracy_tensor.eval(feed_dict={data_placeholder: test_data, labels_placeholder: test_labels})
        precision = precision_tensor.eval(feed_dict={data_placeholder: test_data, labels_placeholder: test_labels})/2
        recall    = recall_tensor.eval(feed_dict={data_placeholder: test_data, labels_placeholder: test_labels})/2
        auroc     = auroc_tensor.eval(feed_dict={data_placeholder: test_data, labels_placeholder: test_labels})
        aupr      = aupr_tensor.eval(feed_dict={data_placeholder: test_data, labels_placeholder: test_labels})
        # classification_accuracy = runner.evaluate_model(accuracy_tensor, batch_size,
        #                                                 feed_vars=(data_placeholder, labels_placeholder),
        #                                                 feed_data=pt.train.feed_numpy(test_batch_num, test_data, test_labels),
        #                                                 print_every=500)
        f1 = calculate_f1(precision, recall)
        if best_f1 < f1:
            best_f1 = f1
            update_best_results(accuracy, precision, recall, f1, aupr, auroc, epoch, best_f1_results)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            update_best_results(accuracy, precision, recall, f1, aupr, auroc, epoch, best_accuracy_results)
        print('Accuracy after %d epoch %g%%' % (epoch + 1, accuracy*100))
        print('F1 after %d epoch %g%%' % (epoch + 1, f1*100))
    print('Train size is {}.Best accuracy is: {}'.format(num_lab, best_accuracy_results))
    print('Train size is {}.Best f1 is: {}'.format(num_lab, best_f1_results))
    print('==================================')
end = time.time()
print("elapse: ", (end-start)/3600)