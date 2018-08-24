# encoding: utf-8

"""
@Author: Kaiqi Yuan
@Since: 2018-08-11 上午12:34
@Description: 将所有的特征放直接放到lstm分类器中
"""

import sys
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/BaseHierarchyContext'])
import time
from BaseHierarchyContext.dataset_process import load_context
from BaseHierarchy.dataset_process import load_dataset
from utils import permute_dataset, fully_connected_layer, convolutional_layer, lenet5, BiRNN, next_batch_data_hierarchy, \
    calculate_f1, update_best_results
import tensorflow as tf
import prettytensor as pt

batch_size = 40
epochs = 1200  # Number of epochs through the full dataset
learning_rate = 3e-3  # 原来的学习率为3e-4
num_lab = 28800
dim_y = 2
latent_dim = 500
classify_dataset_path = "/data/cdy/ykq/BaseWordvecHierarchyDataset"
hierarchy_types = 10 # deepwalk, node2vec, LINE, TransE, PTransE

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


def reshape_data(tensor, per_example_length=1):
  """Reshapes input so that it is appropriate for sequence_lstm..

  The expected format for sequence lstms is
  [timesteps * batch, per_example_length] and the data produced by the utilities
  is [batch, timestep, *optional* expected_length].  The result can be cleaved
  so that there is a Tensor per timestep.

  Args:
    tensor: The tensor to reshape.
    per_example_length: The number of examples at each timestep.
  Returns:
    A Pretty Tensor that is compatible with cleave and then sequence_lstm.

  """
  # We can put the data into a format that can be easily cleaved by
  # transposing it (so that it varies fastest in batch) and then making each
  # component have a single value.
  # This will make it compatible with the Pretty Tensor function
  # cleave_sequence.
  dims = [1, 0]
  print("tensor.get_shape().ndims: ", tensor.get_shape().ndims)
  for i in range(2, tensor.get_shape().ndims):
    dims.append(i)
  print("dims", dims)
  print("tf.transpose(tensor, dims): ", tf.transpose(tensor, dims))
  return pt.wrap(tf.transpose(tensor, dims)).reshape([-1, per_example_length]) # -1表示不知道行数有多少

def lstm_classifier(labels, x, timesteps, per_example_length, phase=pt.Phase.train):
    x = pt.wrap(tf.transpose(x, [1, 0, 2])).reshape([-1, per_example_length])
    # labels = reshape_data(tf.reshape(tf.tile(labels, [1, time_steps]), [batch_size, time_steps, 2]),
    #                       per_example_length=2)
    # input = pt.wrap(x)# tf.transpose(x, [1, 0, 2])).reshape([-1, per_example_length])
    print("x shape:", x.shape)
    print("label shape:", labels.shape)
    lstm = x.cleave_sequence(timesteps).sequence_lstm(500).sequence_lstm(100)
    return (lstm.squash_sequence()
            .dropout(keep_prob=0.8, phase=phase)
            .fully_connected(40, activation_fn=None).softmax_classifier(2, labels))


train_data, train_hierarchy, train_labels, test_data, test_hierarchy, test_labels = load_dataset(classify_dataset_path, num_lab//2)
hierarchy_dim = train_hierarchy.shape[1] // hierarchy_types
dim_x = train_data.shape[1]
feature_dim = int(dim_x // 2)
train_batch_num = train_data.shape[0] // batch_size
test_batch_num = test_data.shape[0] // batch_size
train_head_data    = train_data[:, 0:feature_dim]
train_tail_data    = train_data[:, feature_dim:]
test_head_data     = test_data[:, 0:feature_dim]
test_tail_data     = test_data[:, feature_dim:]
# print("dim_x: ", dim_x, "\ntrain_size:", train_data.shape[0], "\ntest_size:",test_data.shape[0])
print("train_head_data: ", train_head_data.shape)
print("train_tail_data: ", train_tail_data.shape)
print("test_head_data: ",  test_head_data.shape)
print("test_tail_data: ",  test_tail_data.shape)
print("hierarchy dim: ", hierarchy_dim)


train_head_deepwalk, train_head_node2vec, train_head_LINE, train_head_Trans, train_head_PTransE, \
train_tail_deepwalk, train_tail_node2vec, train_tail_LINE, train_tail_Trans, train_tail_PTransE = load_hierarchy(train_hierarchy)
test_head_deepwalk, test_head_node2vec, test_head_LINE, test_head_Trans, test_head_PTransE, \
test_tail_deepwalk, test_tail_node2vec, test_tail_LINE, test_tail_Trans, test_tail_PTransE= load_hierarchy(test_hierarchy)

time_steps = 62
vector_dim = 211

head_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
head_hierarchy_placeholder = tf.placeholder(tf.float32, [batch_size, hierarchy_dim])
tail_data_placeholder = tf.placeholder(tf.float32, [batch_size, feature_dim])
tail_hierarchy_placeholder = tf.placeholder(tf.float32, [batch_size, hierarchy_dim])
labels_placeholder    = tf.placeholder(tf.float32, [batch_size, dim_y])
labels_input = reshape_data(tf.reshape(tf.tile(labels_placeholder, [1, time_steps]), [batch_size, time_steps, 2]),
                          per_example_length=2)
print("labels input:", labels_input.shape)

input_data = tf.reshape(tf.concat([head_data_placeholder, head_hierarchy_placeholder,
                                   tail_data_placeholder, tail_hierarchy_placeholder], 1),
                        [batch_size, time_steps, vector_dim])
print("input_data shape: ", input_data.shape)

with tf.variable_scope("drug_classifier"):
    train_result = lstm_classifier(labels_input, input_data, time_steps, vector_dim)
optimizer = tf.train.AdagradOptimizer(0.5)
train_op = pt.apply_optimizer(optimizer, losses=[train_result.loss])

with tf.variable_scope("drug_classifier", reuse=True):
    result = lstm_classifier(labels_input, input_data, time_steps, vector_dim, phase=pt.Phase.test)
pred_logit = result.softmax.softmax_activation()
accuracy_tensor = result.softmax.evaluate_classifier(labels_input, phase=pt.Phase.test)
precision_tensor, recall_tensor = result.softmax.evaluate_precision_recall(labels_input, phase=pt.Phase.test)
_, auroc_tensor = tf.metrics.auc(labels_input, pred_logit)
_, aupr_tensor  = tf.metrics.auc(labels_input, pred_logit, curve="PR")

start = time.time()
train_head_hierarchy = train_head_deepwalk
train_tail_hierarchy = train_tail_deepwalk
test_head_hierarchy  = test_head_deepwalk
test_tail_hierarchy  = test_tail_deepwalk

best_f1 = 0
best_accuracy = 0
best_f1_results = {"precision": 0, "recall": 0, "auroc": 0, "aupr": 0, "accuracy": 0, "epoch": 0, "f1":0}
best_accuracy_results = {"precision": 0, "recall": 0, "auroc": 0, "aupr": 0, "accuracy": 0, "epoch": 0, "f1":0}

runner = pt.train.Runner()
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        train_head_data, train_head_hierarchy,\
        train_tail_data, train_tail_hierarchy, \
        train_labels = permute_dataset((train_head_data, train_head_hierarchy,
                                        train_tail_data, train_tail_hierarchy,
                                        train_labels))

        runner.train_model(train_op, result.loss, train_batch_num,
                           feed_vars=(head_data_placeholder, head_hierarchy_placeholder,
                                      tail_data_placeholder, tail_hierarchy_placeholder,
                                      labels_placeholder),
                           feed_data=pt.train.feed_numpy(batch_size,
                                                         train_head_data, train_head_hierarchy,
                                                         train_tail_data, train_tail_hierarchy,
                                                         train_labels),
                           print_every=500)
        accuracy = 0
        precision = 0
        recall = 0
        auroc = 0
        aupr  = 0
        num_batches = test_labels.shape[0]/batch_size
        for test_batch_head_data, test_batch_head_hierarchy,\
            test_batch_tail_data, test_batch_tail_hierarchy,\
            test_batch_labels in next_batch_data_hierarchy(test_head_data, test_head_hierarchy,\
                                            test_tail_data, test_tail_hierarchy,\
                                            test_labels, batch_size):
            accuracy += accuracy_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                        head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                        tail_data_placeholder:      test_batch_tail_data,
                                                        tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                        labels_placeholder:         test_batch_labels})
            precision += precision_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                          head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                          tail_data_placeholder:      test_batch_tail_data,
                                                          tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                          labels_placeholder:         test_batch_labels}) / 2
            recall += recall_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                    head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                    tail_data_placeholder:      test_batch_tail_data,
                                                    tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                    labels_placeholder:         test_batch_labels}) / 2
            auroc += auroc_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                  head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                  tail_data_placeholder:      test_batch_tail_data,
                                                  tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                  labels_placeholder:         test_batch_labels})
            aupr += aupr_tensor.eval(feed_dict={head_data_placeholder:      test_batch_head_data,
                                                head_hierarchy_placeholder: test_batch_head_hierarchy,
                                                tail_data_placeholder:      test_batch_tail_data,
                                                tail_hierarchy_placeholder: test_batch_tail_hierarchy,
                                                labels_placeholder:         test_batch_labels})

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