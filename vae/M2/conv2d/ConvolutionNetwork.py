# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: ConvolutionNetwork.py
@time: 18-6-5 下午3:50
@description:
"""

import prettytensor as pt
import tensorflow as tf
import numpy as np

class ConvolutionallyConnected(object):
    def __init__(self,
                 dim_output,
                 hidden_layers=[500]):
        self.dim_output = dim_output
        self.hidden_layers = hidden_layers

    # 为了公用一些函数，但是不同阶段训练的node数目不一样，用pt.Phase来区别
    def output(self, inputs, phase=pt.Phase.train):
        # print("input_size:", inputs.shape)
        input_size = inputs.shape[0]
        input_dim = inputs.shape[1]
        # inputs = np.reshape(inputs, [input_size, input_dim, 1, 1])
        inputs = tf.reshape(inputs, [input_size, input_dim, 1, 1])
        inputs = pt.wrap(inputs)
        with pt.defaults_scope(phase=phase, activation_fn=tf.nn.relu, l2loss=0.00001):
            inputs = inputs.conv2d(5, 20).max_pool(2, 2).conv2d(5, 50).max_pool(2, 2).flatten()
            # print(self.hidden_layers)
            for layer in self.hidden_layers:
                inputs = inputs.fully_connected(layer)

            # A Pretty Tensor handle to the layer.
            outputs = inputs.fully_connected(self.dim_output, activation_fn=None)
            # print("output: ", outputs.shape)
            return outputs
