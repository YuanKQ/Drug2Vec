# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: utils.py
@time: 18-6-13 上午8:35
@description:
"""
import pickle
import random
import numpy as np
import prettytensor as pt
import tensorflow as tf


def build_drug_feature_matrix(ddi_file, file_prefix, targetfile):
    drugs_list = load_drugs_list(ddi_file)
    feature_matrix_dict = {}
    feature_matrix_dict["actionCode"]= load_feature_matrix(file_prefix + "drug_actionCode_matrix_dict.pickle")
    feature_matrix_dict["atc"]       = load_feature_matrix(file_prefix + "drug_atc_matrix_dict.pickle")
    feature_matrix_dict["MACCS"]     = load_feature_matrix(file_prefix + "drug_MACCS_matrix_dict.pickle")
    feature_matrix_dict["SIDER"]     = load_feature_matrix(file_prefix + "drug_SIDER_matrix_dict.pickle")
    feature_matrix_dict["phyCode"]   = load_feature_matrix(file_prefix + "drug_phyCode_matrix_dict.pickle")
    feature_matrix_dict["target"]    = load_feature_matrix(file_prefix + "drug_target_matrix_dict.pickle")
    feature_matrix_dict["word2vec"]  = load_feature_matrix(file_prefix + "drug_word2vec_matrix_dict.pickle")
    feature_matrix_dict["deepwalk"]  = load_feature_matrix(file_prefix + "drug_deepwalk_matrix_dict.pickle")
    feature_matrix_dict["LINE"]      = load_feature_matrix(file_prefix + "drug_LINE_matrix_dict.pickle")
    feature_matrix_dict["node2vec"]  = load_feature_matrix(file_prefix + "drug_node2vec_matrix_dict.pickle")
    drug_features_dict = {}
    for drug in drugs_list:
        drug_features_dict[drug] = {}
        for key in feature_matrix_dict.keys():
            drug_features_dict[drug][key] = feature_matrix_dict[key][drug]
    with open(targetfile, "wb") as wf:
        pickle.dump(drug_features_dict, wf)


def load_drugs_list(ddi_file):
    with open(ddi_file, 'rb') as rf:
        drugs_list = pickle.load(rf)
    return drugs_list


def load_feature_matrix(file):
    with open(file, 'rb') as rf:
        feature_dict = pickle.load(rf)
    return feature_dict


def shuffle_feature_data(relations, drug_all_dict):
    idx = list(range(len(relations)))
    random.seed(10)
    random.shuffle(idx)
    feature_matrix = []
    lab_matrix = []
    for i in idx:
        head = relations[i][0]
        tail = relations[i][1]
        rel = relations[i][2]
        matrix = np.concatenate((drug_all_dict[head], drug_all_dict[tail]))
        feature_matrix.append(matrix)
        matrix1 = np.concatenate((drug_all_dict[tail], drug_all_dict[head]))
        feature_matrix.append(matrix1)
        if rel == "increase":
            lab = np.array([0,1])
        else:
            lab = np.array([1, 0])
        lab_matrix.append(lab)
        lab_matrix.append(lab)
    return feature_matrix, lab_matrix


def shuffle_feature_hierarchy_data(relations, drug_all_dict, network_dict):
    idx = list(range(len(relations)))
    # 保证每一次shuffle都能够得到相同的随机数列
    random.seed(10)
    random.shuffle(idx)
    feature_matrix = []
    lab_matrix = []
    network_matrix = []
    for i in idx:
        head = relations[i][0]
        tail = relations[i][1]
        rel = relations[i][2]
        feature_matrix.append(np.concatenate((drug_all_dict[head], drug_all_dict[tail])))
        network_matrix.append(np.concatenate((network_dict[head], network_dict[tail])))
        feature_matrix.append(np.concatenate((drug_all_dict[tail], drug_all_dict[head])))
        network_matrix.append(np.concatenate((network_dict[tail], network_dict[head])))
        if rel == "increase":
            lab = np.array([0,1])
        else:
            lab = np.array([1, 0])
        lab_matrix.append(lab)
        lab_matrix.append(lab)
    print("feature_matrix: ", len(feature_matrix), len(feature_matrix[0]))
    print("network_matrix: ", len(network_matrix), len(network_matrix[0]))
    print("lab_matrix: ", len(lab_matrix), len(lab_matrix[0]))
    return feature_matrix, network_matrix, lab_matrix


def dump_dataset(feature_matrix, lab_matrix, target_file_prefix):
    partition = 5000
    start = 0
    print(target_file_prefix)
    # for i in range(count):
    for i in range(3, 6):
        end = (i+1) * partition
        with open("%s_%d.pickle"%(target_file_prefix, i), "wb") as wf:
            pickle.dump(feature_matrix[start:end], wf)
            pickle.dump(lab_matrix[start:end], wf)
        print("start: %d, end: %d" % (start, end))
        start = end


def permute_dataset(arrays):
    """Permute multiple numpy arrays with the same order."""
    if any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError('All arrays must be the same length.')
    random_state = np.random
    order = random_state.permutation(len(arrays[0]))
    return [a[order] for a in arrays]


def load_dataset_split(path, n_labeled, start=0, sample_size=3, valid_test_ratio=50):
    decrease_feature_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_labs = []
    for i in range(start, start+sample_size):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase：", len(increase_feature_matrix), len(increase_feature_matrix[0]), len(increase_labs),
          len(increase_labs[0]))
    for i in range(start, start+sample_size):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i),
                'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))
    print("decrease：", len(decrease_feature_matrix), len(decrease_feature_matrix[0]), len(decrease_labs),
          len(decrease_labs[0]))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count / valid_test_ratio)
    train_count = sample_count - valid_count - test_count
    print("util/traincount: ", train_count)

    n_labeled_perclass = int(n_labeled/len(increase_labs))
    x_label = decrease_feature_matrix[0: n_labeled]
    x_label.extend(increase_feature_matrix[0: n_labeled])
    y_label = decrease_labs[0: n_labeled]
    y_label.extend(increase_labs[0: n_labeled])

    # ==============================
    valid_x = decrease_feature_matrix[train_count:train_count+valid_count]
    valid_x.extend(increase_feature_matrix[train_count:train_count+valid_count])
    valid_y = decrease_labs[train_count:train_count+valid_count]
    valid_y.extend(increase_labs[train_count:train_count+valid_count])

    test_x = decrease_feature_matrix[train_count+valid_count:]
    test_x.extend(increase_feature_matrix[train_count+valid_count:])
    test_y = decrease_labs[train_count+valid_count:]
    test_y.extend(increase_labs[train_count+valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_label), np.array(y_label), np.array(valid_x), \
           np.array(valid_y), np.array(test_x), np.array(test_y)

def load_dataset(path, start=0,sample_size=3, valid_test_ratio=50):
    decrease_feature_matrix = []
    decrease_labs = []
    increase_feature_matrix = []
    increase_labs = []
    for i in range(start, start + sample_size):
        with open(
                "%s/increase_features_labs_matrix_%d.pickle" % (path, i), 'rb') as rf:
            increase_feature_matrix.extend(pickle.load(rf))
            increase_labs.extend(pickle.load(rf))
    print("increase：", len(increase_feature_matrix), len(increase_feature_matrix[0]), len(increase_labs),
          len(increase_labs[0]))
    for i in range(start, start + sample_size):
        with open(
                "%s/decrease_features_labs_matrix_%d.pickle" % (path, i), 'rb') as rf:
            decrease_feature_matrix.extend(pickle.load(rf))
            decrease_labs.extend(pickle.load(rf))
    print("decrease：", len(decrease_feature_matrix), len(decrease_feature_matrix[0]), len(decrease_labs),
          len(decrease_labs[0]))

    sample_count = len(increase_labs)
    valid_count = test_count = int(sample_count/valid_test_ratio)
    train_count = sample_count - valid_count - test_count

    x_train = decrease_feature_matrix[0: train_count]
    x_train.extend(increase_feature_matrix[0: train_count])
    y_train = decrease_labs[0: train_count]
    y_train.extend(increase_labs[0:train_count])

    # ==============================
    valid_x = decrease_feature_matrix[train_count:train_count+valid_count]
    valid_x.extend(increase_feature_matrix[train_count:train_count+valid_count])
    valid_y = decrease_labs[train_count:train_count+valid_count]
    valid_y.extend(increase_labs[train_count:train_count+valid_count])

    test_x = decrease_feature_matrix[train_count+valid_count:]
    test_x.extend(increase_feature_matrix[train_count+valid_count:])
    test_y = decrease_labs[train_count+valid_count:]
    test_y.extend(increase_labs[train_count+valid_count:])
    print("========Finish loading train_dataset============")
    return np.array(x_train), np.array(y_train), np.array(valid_x), \
           np.array(valid_y), np.array(test_x), np.array(test_y)

def fully_connected_layer(input, dim_output):
    input = pt.wrap(input)
    with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
        return input.fully_connected(dim_output)

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
        return (images.conv2d(5, 20).max_pool(2, 2).conv2d(5, 20).max_pool(2, 2)
                .flatten().fully_connected(500).softmax_classifier(dim_y, labels))


def convolutional_layer(input, dim_output, dim_z=1):
    size = input.shape[0]
    dim = input.shape[1]
    input = tf.reshape(input, [size, dim, 1, dim_z])
    input = pt.wrap(input)
    with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
        return input.conv2d(5, 20).max_pool(2, 2).flatten().fully_connected(dim_output)


# def BiRNN(x, time_steps, num_hidden, weights, biases):
#     # head_context_trans = BiRNN(head_context_placeholder, 100, 128, weights, biases)
#
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, timesteps, n_input)
#     # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)
#
#     # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
#     print("before stack: x.stack", x.shape)
#     x = tf.unstack(x, time_steps, 1)
#     tf.Session().run(x)
#     print("after stack: ", x.shape)
#
#     # Define lstm cells with tensorflow
#     # Forward direction cell
#     # tf.reset_default_graph()
#     with tf.Graph().as_default():
#         lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#         lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
#         # with tf.variable_scope('forward'):
#         #     lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#         # # Backward direction cell
#         # with tf.variable_scope('forward'):
#         #     lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
#         # Get lstm cell output
#         # ValueError: Shape must be rank 2 but is rank 3 for 'MatMul' (op: 'MatMul') with input shapes: [2,40,128], [256,100]
#         try:
#             outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                                          dtype=tf.float32)
#         except Exception: # Old TensorFlow version only returns outputs not states
#             outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                                    dtype=tf.float32)
#
#         # Linear activation, using rnn inner loop last output
#         print("outputs.shape", np.array(outputs).shape)
#     return tf.nn.softmax(tf.matmul(outputs[-1], weights['out']) + biases['out'])

def BiRNN(x, time_steps, per_example_length, num_hidden):
    #      sentence_len, word2vec_dim,       num_hidden, clatent_dim
    # x.shape = [batch_size, sentence_len, word2vec_dim]
    batch_size = x.shape[0]
    input = pt.wrap(tf.transpose(x, [1, 0, 2])).reshape([-1, per_example_length])
    print("input.shape:", input.shape)
    with tf.variable_scope('context_lstm', reuse=tf.AUTO_REUSE):
        lstm = input.cleave_sequence(time_steps).sequence_lstm(num_hidden).squash_sequence().dropout(0.7).fully_connected(1, activation_fn=tf.nn.relu)
        lstm = lstm.reshape([batch_size, -1])
        print("final lstm:", lstm.shape)
        return lstm

def calculate_f1(precision, recall):
    return 2*precision*recall/(precision + recall)

def update_best_results(accuracy, precision, recall, f1, aupr, auroc, epoch, best_results):
    best_results["precision"] = precision
    best_results["recall"] = recall
    best_results["accuracy"] = accuracy
    best_results["f1"] = f1
    best_results["aupr"] = aupr
    best_results["auroc"] = auroc
    best_results["epoch"] = epoch


def next_batch(test_head_data, test_head_hierarchy, test_head_context, \
               test_tail_data, test_tail_hierarchy, test_tail_context, \
               test_labels, batch_size):
    size = test_labels.shape[0]
    count = size // batch_size
    # print("count: ", count)
    for i in range(count):
        start = i * batch_size
        end = start + batch_size
        yield [test_head_data[start: end], test_head_hierarchy[start: end], test_head_context[start: end], \
               test_tail_data[start: end], test_tail_hierarchy[start: end], test_tail_context[start: end], \
               test_labels[start: end]]


def next_batch_data_hierarchy(test_head_data, test_head_hierarchy, test_tail_data, test_tail_hierarchy, test_labels, batch_size):
    size = test_labels.shape[0]
    count = size // batch_size
    # print("count: ", count)
    for i in range(count):
        start = i * batch_size
        end = start + batch_size
        yield [test_head_data[start: end], test_head_hierarchy[start: end], test_tail_data[start: end], test_tail_hierarchy[start: end], test_labels[start: end]]

if __name__ == '__main__':
    build_drug_feature_matrix("drugs_ddi_v5.pickle", "", "drug_features_dict_v5.pickle")
    print("end")