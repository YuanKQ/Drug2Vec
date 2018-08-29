# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from description import *

class Settings(object):
    def __init__(self):
        self.structure_entity_vocab_size = 188156
        self.structure_relation_vocab_size = 43990
        self.structure_embedding_size = 50

        self.text_relation_vocab_size = 43990
        self.text_embedding_size = 50

        self.word_vocab_size = 50615
        self.word_embedding_size = 50

        self.path_length = 6

        self.num_epochs = 51
        self.num_classes = 2
        self.batch_size = 32

        self.path_vector_size = 230
        self.des_vector_size = 230

        self.description_vector_size = self.text_embedding_size

        self.keep_prob = 0.5

        self.num_layers = 1
        self.desciption_length = 60


class Model:
    def __init__(self, is_training, settings):

        self.path_length = path_length = settings.path_length
        self.structure_entity_vocab_size = structure_entity_vocab_size = settings.structure_entity_vocab_size
        self.structure_relation_vocab_size = structure_relation_vocab_size = settings.structure_relation_vocab_size
        self.structure_embedding_size = structure_embedding_size = settings.structure_embedding_size
        self.desciption_length = desciption_length = settings.desciption_length
        self.text_relation_vocab_size = text_relation_vocab_size = settings.text_relation_vocab_size
        self.num_classes = num_classes = settings.num_classes
        self.path_vector_size = path_vector_size = settings.path_vector_size
        self.des_vector_size = des_vector_size = settings.des_vector_size
        self.description_vector_size = description_vector_size = settings.description_vector_size
        self.batch_size = batch_size = settings.batch_size
        self.text_embedding_size = text_embedding_size = settings.text_embedding_size
        self.word_embedding_size = word_embedding_size = settings.word_embedding_size
        self.word_vocab_size = word_vocab_size = settings.word_vocab_size


        # 输入数据
        self.input_description_seq = tf.placeholder(dtype=tf.int32, shape=[None, path_length], name='input_description_seq')
        self.input_entity_seq = tf.placeholder(dtype=tf.int32, shape=[None, path_length], name='input_entity_sequence')
        self.input_relation_seq = tf.placeholder(dtype=tf.int32, shape=[None, path_length], name='input_relation_sequence')
        self.descriptions = tf.placeholder(dtype=tf.int32, shape=[None, desciption_length], name='descriptions')
        self.total_description = tf.placeholder(dtype=tf.int32, shape=[1], name='total_description')[0]
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[batch_size + 1], name='total_shape')
        total_num = self.total_shape[-1]
        
        # embedding 初始化
        structure_entity_embedding = tf.Variable(tf.constant(0.0, shape=[structure_entity_vocab_size, structure_embedding_size]), trainable=False, name="structure_entity_embedding")   
        structure_relation_embedding = tf.Variable(tf.constant(0.0, shape=[structure_relation_vocab_size, structure_embedding_size]), trainable=False, name="structure_relation_embedding")   
        
        self.structure_entity_embedding_placeholder = tf.placeholder(tf.float32, [structure_entity_vocab_size, structure_embedding_size], name='structure_entity_embedding_placeholder')
        self.structure_relation_embedding_placeholder = tf.placeholder(tf.float32, [structure_relation_vocab_size, structure_embedding_size], name='structure_relation_embedding_placeholder')

        self.structure_entity_embedding_init = structure_entity_embedding.assign(self.structure_entity_embedding_placeholder)
        self.structure_relation_embedding_init = structure_relation_embedding.assign(self.structure_relation_embedding_placeholder)

        #structure_entity_embedding = tf.get_variable(shape=[structure_entity_vocab_size, structure_embedding_size], name="structure_entity_embedding", trainable=False)
        #structure_relation_embedding = tf.get_variable(shape=[structure_relation_vocab_size, structure_embedding_size], name="structure_relation_embedding", trainable=False)
            
        word_embedding = tf.Variable(tf.constant(0.0, shape=[word_vocab_size, word_embedding_size]), trainable=False, name="word_embedding")   
        self.word_embedding_placeholder = tf.placeholder(tf.float32, [word_vocab_size, word_embedding_size], name='word_embedding_placeholder')
        self.word_embedding_init = word_embedding.assign(self.word_embedding_placeholder)
            
                    
        # description 文本信息 循环神经网络定义
        des_cell_forward = tf.contrib.rnn.BasicLSTMCell(description_vector_size)
        des_cell_backward = tf.contrib.rnn.BasicLSTMCell(description_vector_size)

        # path 循环神经网络定义
        path_cell_forward = tf.contrib.rnn.GRUCell(path_vector_size)
        path_cell_backward = tf.contrib.rnn.GRUCell(path_vector_size)

        # entity attention
        entity_a = tf.get_variable('entity_attention_A', [structure_embedding_size])
        entity_r = tf.get_variable('entity_query_r', [structure_embedding_size, 1])

        # 全连接层定义
        path_a = tf.get_variable('attention_A', [path_vector_size])
        path_r = tf.get_variable('query_r', [path_vector_size, 1])
        attention_w = tf.get_variable('attention_w', [description_vector_size, 1])
        relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, path_vector_size])
        path_d = tf.get_variable('bias_d', [self.num_classes])

        # 映射矩阵
        structure_map = tf.get_variable('structure_map', [structure_embedding_size,structure_embedding_size])
        text_map = tf.get_variable('text_map',[text_embedding_size,text_embedding_size])


        """
        :param is_training: 是否要进行训练.如果is_training=False,则不会进行参数的修正。
        """
        if is_training and settings.keep_prob < 1:
            path_cell_forward = tf.contrib.rnn.DropoutWrapper(path_cell_forward, output_keep_prob=settings.keep_prob)
            path_cell_backward = tf.contrib.rnn.DropoutWrapper(path_cell_backward, output_keep_prob=settings.keep_prob)

            des_cell_forward = tf.contrib.rnn.DropoutWrapper(des_cell_forward, output_keep_prob=settings.keep_prob)
            des_cell_backward = tf.contrib.rnn.DropoutWrapper(des_cell_backward, output_keep_prob=settings.keep_prob)


        path_cell_forward = tf.contrib.rnn.MultiRNNCell([path_cell_forward] * settings.num_layers)
        path_cell_backward = tf.contrib.rnn.MultiRNNCell([path_cell_backward] * settings.num_layers)

        des_cell_forward = tf.contrib.rnn.MultiRNNCell([des_cell_forward] * settings.num_layers)
        des_cell_backward = tf.contrib.rnn.MultiRNNCell([des_cell_backward] * settings.num_layers)

        self._des_initial_state_forward = des_cell_forward.zero_state(self.total_description, tf.float32)
        self._des_initial_state_backward = des_cell_backward.zero_state(self.total_description, tf.float32)

        # 读取词向量
        des_inputs_forward = tf.concat([tf.nn.embedding_lookup(word_embedding, self.descriptions)], 2)
        des_inputs_backward = tf.concat([tf.nn.embedding_lookup(word_embedding, tf.reverse(self.descriptions, [False, True]))], 2)

        # entity description
        des_outputs_forward = []
        
        # Bi-LSTM layer
        des_state_forward = self._des_initial_state_forward
        with tf.variable_scope('LSTM_FORWARD'):
            for step in range(desciption_length):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                (des_cell_output_forward, des_state_forward) = des_cell_forward(des_inputs_forward[:, step, :], des_state_forward)
                des_outputs_forward.append(des_cell_output_forward)

        des_outputs_backward = []

        des_state_backward = self._des_initial_state_backward
        with tf.variable_scope('LSTM_BACKWARD'):
            for step in range(desciption_length):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                (des_cell_output_backward, des_state_backward) = des_cell_backward(des_inputs_backward[:, step, :], des_state_backward)
                des_outputs_backward.append(des_cell_output_backward)

        des_output_forward = tf.reshape(tf.concat(des_outputs_forward, 1), [self.total_description, desciption_length, description_vector_size])
        des_output_backward = tf.reverse(tf.reshape(tf.concat(des_outputs_backward, 1), [self.total_description, desciption_length, description_vector_size]),
                                     [1])  # [False, True, False])

        # word-level attention layer
        des_output_h = tf.add(des_output_forward, des_output_backward)
        des_embedding = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(des_output_h), [self.total_description * desciption_length, description_vector_size]), attention_w),
                       [self.total_description, desciption_length])), [self.total_description, 1, desciption_length]), des_output_h), [self.total_description, description_vector_size])


        path_repre = []
        self.path_alpha = []
        path_s = []

        path_out = []

        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0

        self._path_initial_state_forward = path_cell_forward.zero_state(total_num, tf.float32)
        self._path_initial_state_backward = path_cell_backward.zero_state(total_num, tf.float32)


        # embedding layer       准备输入的数据
        entity_des_path_inputs = tf.reshape(tf.matmul(tf.reshape(tf.nn.embedding_lookup(des_embedding, self.input_description_seq), [total_num*path_length, text_embedding_size]),text_map),[total_num*path_length, 1, text_embedding_size])
        entity_stru_path_inputs = tf.reshape(tf.matmul(tf.reshape(tf.nn.embedding_lookup(structure_entity_embedding, self.input_entity_seq), [total_num*path_length, structure_embedding_size]),structure_map),[total_num*path_length, 1, structure_embedding_size])
        # entity_path_inputs = []
        # for i in range(total_num*path_length):
        entity_repre = tf.reshape(tf.concat([entity_des_path_inputs,entity_stru_path_inputs],1),[total_num*path_length*2, structure_embedding_size])

        # wds1 = tf.multiply(entity_repre, entity_a)
        wds2 = tf.matmul(entity_repre, entity_r)
        # wds2 = tf.scan(lambda a, x: tf.matmul(x, entity_r), entity_repre)
        wds3 = tf.nn.softmax(tf.reshape(wds2, [total_num*path_length,2]))
        entity_alpha = tf.reshape(wds3, [total_num*path_length,1, 2])
        entity_path_inputs = tf.reshape(tf.matmul(entity_alpha, tf.reshape(entity_repre,[total_num*path_length,2, structure_embedding_size])), [total_num,path_length,structure_embedding_size])
            


        relation_path_inputs =  tf.reshape(tf.matmul(tf.reshape(tf.nn.embedding_lookup(structure_relation_embedding, self.input_relation_seq), [total_num*path_length, structure_embedding_size]),structure_map),[total_num, path_length, structure_embedding_size])
        
        path_outputs_forward = []

        path_state_forward = self._path_initial_state_forward

        # Bi-GRU layer
        with tf.variable_scope('GRU_FORWARD'):
            for step in range(path_length*2):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                gru_input = None
                if step % 2 == 0:
                    i = int(step / 2)
                    gru_input = entity_path_inputs[:,i,:]
                else:
                    i = int((step-1) / 2)
                    gru_input = relation_path_inputs[:,i,:]

                (cell_output_forward, path_state_forward) = path_cell_forward(gru_input, path_state_forward)
                path_outputs_forward.append(cell_output_forward)

        path_outputs_backward = []

        path_state_backward = self._path_initial_state_backward
        with tf.variable_scope('GRU_BACKWARD'):
            for step in range(path_length*2):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                gru_input = None
                if step % 2 == 0:
                    i = path_length-int(step / 2)-1
                    gru_input = entity_path_inputs[:,i,:]
                else:
                    i = path_length-int((step-1) / 2)-1
                    gru_input = relation_path_inputs[:,i,:]

                (cell_output_backward, path_state_backward) = path_cell_backward(gru_input, path_state_backward)
                path_outputs_backward.append(cell_output_backward)

        path_output_forward = tf.reshape(tf.concat(path_outputs_forward, 1), [total_num, path_length*2, path_vector_size])
        path_output_backward = tf.reverse(tf.reshape(tf.concat(path_outputs_backward, 1), [total_num, path_length*2, path_vector_size]),
                                     [1])  # [False, True, False])

        # word-level attention layer
        path_output_h = tf.add(path_output_forward, path_output_backward)[::,-1,::]
        # attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
        #     tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [total_num * num_steps, gru_size]), attention_w),
        #                [total_num, num_steps])), [total_num, 1, num_steps]), output_h), [total_num, gru_size])


        # sentence-level attention layer
        for i in range(batch_size):

            path_repre.append(tf.tanh(path_output_h[self.total_shape[i]:self.total_shape[i + 1]]))
            path_batch_size = self.total_shape[i + 1] - self.total_shape[i]

            self.path_alpha.append(
                tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(path_repre[i], path_a), path_r), [path_batch_size])),
                           [1, path_batch_size]))

            path_s.append(tf.reshape(tf.matmul(self.path_alpha[i], path_repre[i]), [path_vector_size, 1]))
            path_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, path_s[i]), [self.num_classes]), path_d))

            self.prob.append(tf.nn.softmax(path_out[i]))

            with tf.name_scope("output"):
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):
                self.loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=path_out[i], labels=self.input_y[i])))
                # self.loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_out[i], labels=self.input_y[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            with tf.name_scope("accuracy"):
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 0)), "float"),
                                   name="accuracy"))

class Description_Path(object):
    def __init__(self, settings):
        self.e2id = {}
        self.id2e = {}
        self.path_len = settings.path_length
        self.id = 0
        self.desciption_length = settings.desciption_length

    def add(self,path):
        for i in range(self.path_len):
            entity = int(path[i])

            if entity in self.e2id:
                pass
            else:
                self.e2id[entity] = self.id
                self.id2e[self.id] = entity

                self.id += 1

    def get_path(self, path):
        new_path = []
        for i in range(self.path_len):
            entity = int(path[i])
            id = self.e2id[entity]
            new_path.append(id)

        return np.array(new_path)

    def get_decription(self, entity_des):
        description = []

        for i in range(self.id):
            description.append(entity_des.get_des(self.id2e[i]))

        return np.array(description)
