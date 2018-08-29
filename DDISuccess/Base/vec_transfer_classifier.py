# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: vec_transfer_classifier.py
@time: 18-6-13 下午9:25
@description: 药物特征（flatten为一维）经过处理后再拼接在一起输入到conv分类器中
<<<<<<< HEAD
 - 矩阵变换后: start with 65.667, end with 82, best is 86.5
 - 加上衡量数据的离散程度: start with 61.8333
 - 加入距离公式： start with 66.3333
=======
 - 全连接网络变换后: 63.666
 - 加入msr
======================
 - 改成全连接网络后： 66.666
"""
import sys
import time

sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/Base'])

from utils import load_dataset_split, permute_dataset, fully_connected_layer, convolutional_layer, lenet5
import tensorflow as tf
import prettytensor as pt

batch_size = 40
epochs = 800  # Number of epochs through the full dataset
learning_rate = 3e-3  # 原来的学习率为3e-4
num_lab = 24800
dim_y = 2
latent_dim = 500
classify_dataset_path = "/data/cdy/ykq/BaseDataset/TestDataset"

train_data, train_labels, _, __, test_data, test_labels = load_dataset_split(classify_dataset_path, num_lab//2, 3)
dim_x = train_data.shape[1]
feature_dim = int(dim_x // 2)
train_batch_num = train_data.shape[0] // batch_size
test_batch_num = test_data.shape[0] // batch_size
test_head_data = test_data[:, 0:feature_dim]
test_tail_data = test_data[:, feature_dim:dim_x]
print("dim_x: ", dim_x, "\ntrain_size:", train_data.shape[0], "\ntest_size:",test_data.shape[0])


head_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
tail_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
labels_placeholder    = tf.placeholder(tf.float32, [batch_size, dim_y])

head_trans = convolutional_layer(head_data_placeholder, latent_dim)
tail_trans = convolutional_layer(tail_data_placeholder, latent_dim)
#head_trans = fully_connected_layer(head_data_placeholder, latent_dim)
#tail_trans = fully_connected_layer(tail_data_placeholder, latent_dim)
# 欧拉距离
euclidean = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(head_trans-tail_trans), axis=1)), [batch_size, 1])
# 余弦距离
head_norm = tf.sqrt(tf.reduce_sum(tf.square(head_trans), axis=1))
tail_norm = tf.sqrt(tf.reduce_sum(tf.square(tail_trans), axis=1))
head_tail_product = tf.reduce_sum(head_trans * tail_trans, axis=1)
cosine = tf.reshape(head_tail_product / (head_norm * tail_norm), [batch_size, 1])

## 加入衡量数据偏离程度
mean = tf.add(head_trans, tail_trans)/2
msr_vec = tf.sqrt(tf.add(tf.square(tf.subtract(head_trans, mean)), tf.square(tf.subtract(tail_trans, mean))))
## 加入距离公式
print("head_trans: ", head_trans.shape)
print("tail_trans: ", tail_trans.shape)
print("msr_vec: ", msr_vec.shape)
input_data = tf.reshape(tf.concat([head_trans, tail_trans, msr_vec], 1), [batch_size, latent_dim*3, 1, 1])

print("input_data shape: ", input_data.shape)
result = lenet5(input_data, labels_placeholder, dim_y)
accuracy = result.softmax.evaluate_classifier(labels_placeholder, phase=pt.Phase.test)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

runner = pt.train.Runner()
best_accuracy = 0
best_epoch = 0
start = time.time()
with tf.Session() as sess:
    for epoch in range(epochs):
        train_data, train_labels = permute_dataset((train_data, train_labels))
        head_train_data = train_data[:, 0:feature_dim]
        tail_train_data = train_data[:, feature_dim:dim_x]
        runner.train_model(train_op, result.loss, train_batch_num,
                           feed_vars=(head_data_placeholder, tail_data_placeholder, labels_placeholder),
                           feed_data=pt.train.feed_numpy(batch_size, head_train_data, tail_train_data, train_labels),
                           print_every=500
                           )

        classification_accuracy = runner.evaluate_model(accuracy, test_batch_num,
                                                        feed_vars=(head_data_placeholder, tail_data_placeholder, labels_placeholder),
                                                        feed_data=pt.train.feed_numpy(batch_size, test_head_data, test_tail_data, test_labels),
                                                        print_every=500)

        # runner.train_model(train_op, result.loss, num_batches,
        #                    feed_vars=(data_placeholder, labels_placeholder),
        #                    feed_data=pt.train.feed_numpy(train_batch_size, train_data, train_labels))
        # classification_accuracy = runner.evaluate_model(accuracy, num_batches,
        #                                                 feed_vars=(data_placeholder, labels_placeholder),
        #                                                 feed_data=pt.train.feed_numpy(test_batch_size, test_data,
        #                                                                               test_labels))

        if best_accuracy < classification_accuracy[0]:
            best_accuracy = classification_accuracy
            best_epoch = epoch
        print('Accuracy after %d epoch %g%%' % (epoch + 1, classification_accuracy[0] * 100))

    print('Train size is {}.Best accuracy is {}%% at {} epoch.'.format(num_lab, best_accuracy, best_epoch))
    print('==================================')
end = time.time()
print("elapse: ", (end-start)/3600)

