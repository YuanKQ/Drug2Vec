# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: base_classifier.py
@time: 18-6-13 下午6:25
@description:
"""
import numpy as np

from utils import load_dataset_split, permute_dataset
import tensorflow as tf
import prettytensor as pt

def lenet5(images, labels, dim_y):
     """Creates a multi layer convolutional network.
     The architecture is similar to that defined in LeNet 5.
     Please change this to experiment with architectures.
     Args:
       images: The input images.
       labels: The labels as dense one-hot vectors.
     Returns:
       A softmax result.
     """
     images = pt.wrap(images)
     with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
        return (images.conv2d(5, 20).max_pool(2, 2).conv2d(5, 50).max_pool(2, 2)
                .flatten().fully_connected(500).softmax_classifier(dim_y, labels))


num_batches = 100  # Number of minibatches in a single epoch
epochs = 800  # Number of epochs through the full dataset
learning_rate = 3e-4
num_lab = 30000
classify_dataset_path = "BaseDataset/test_dataset"
# classify_dataset_path = "/home/yuan/Code/PycharmProjects/vae/ddi/test_dataset"
# train_data, train_labels, _, __, test_data, test_labels = load_dataset(classify_dataset_path, 3)
train_data, train_labels, _u, __u, _, __, test_data, test_labels = load_dataset_split(classify_dataset_path, num_lab//2, 3)
dim_x = train_data.shape[1]
# one_dim = dim_x / 2
# head_feature = train_data[:, 0:one_dim]
# tail_feature = train_data[:, one_dim: dim_x]
# head_trans_placeholder = tf.placeholder(tf.float32, [])

dim_y = 2
train_size = train_data.shape[0]
test_size = test_data.shape[0]
train_data = np.reshape(train_data, [train_size, dim_x, 1, 1])
test_data = np.reshape(test_data, [test_size, dim_x, 1, 1])
assert train_size % num_batches == 0, '#TrainSize % #Batches != 0'
assert test_size % num_batches == 0, '#TestSize % #Batches != 0'
train_batch_size = int(train_size / num_batches)
test_batch_size = int(test_size / num_batches)

data_placeholder = tf.placeholder(tf.float32, [None, dim_x, 1, 1])
labels_placeholder = tf.placeholder(tf.float32, [None, dim_y])

result = lenet5(data_placeholder, labels_placeholder, dim_y)

accuracy = result.softmax.evaluate_classifier(labels_placeholder, phase=pt.Phase.test)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

#save_path = '/data/cdy/ykq/checkpoints/model_conv2d_{}-{}.cpkt'.format(
#            learning_rate, time.strftime("%m-%d-%H%M%S", time.localtime()))
#print("model has been saved: " + save_path)
#runner = pt.train.Runner(save_path)
runner = pt.train.Runner()
best_accuracy = 0
best_epoch = 0
with tf.Session() as sess:
    # print(epochs)
    for epoch in range(epochs):
        train_data, train_labels = permute_dataset((train_data, train_labels))

        # 并没有保存最佳的model
        runner.train_model(train_op, result.loss, num_batches,
                           feed_vars=(data_placeholder, labels_placeholder),
                           feed_data=pt.train.feed_numpy(train_batch_size, train_data, train_labels))
        classification_accuracy = runner.evaluate_model(accuracy, num_batches,
                                                        feed_vars=(data_placeholder, labels_placeholder),
                                                        feed_data=pt.train.feed_numpy(test_batch_size, test_data, test_labels))

        # runner.train_model(
        #     train_op,
        #     result.loss,
        #     EPOCH_SIZE,
        #     feed_vars=(image_placeholder, labels_placeholder),
        #     feed_data=pt.train.feed_numpy(BATCH_SIZE, train_images, train_labels),
        #     print_every=100)
        # classification_accuracy = runner.evaluate_model(
        #     accuracy,
        #     TEST_SIZE,
        #     feed_vars=(image_placeholder, labels_placeholder),
        #     feed_data=pt.train.feed_numpy(BATCH_SIZE, test_images, test_labels))
        if best_accuracy < classification_accuracy[0]:
            best_accuracy = classification_accuracy
            best_epoch = epoch
        print('Accuracy after %d epoch %g%%' % (epoch + 1, classification_accuracy[0]*100))
print('Train size is {}.Best accuracy is {}%% at {} epoch.'.format(num_lab, best_accuracy, best_epoch))
print('==================================')