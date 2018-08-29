# -*- coding: utf-8 -*-
"""
__title__ = 'simple_vae.py'
__IDE__ = 'PyCharm'
__author__ = 'YuanKQ'
__mtime__ = 'Apr 14,2018 22:36'
__mail__ = kq_yuan@outlook.com

__description__==  vae-mnist生成模型
code reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py

详细的代码注释： https://jmetzen.github.io/2015-11-27/vae.html

中文教程： https://zhuanlan.zhihu.com/p/34998569（特别通俗易懂），　https://zhuanlan.zhihu.com/p/23705953（图画得还不错）

"""

from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

# Import MNIST train_dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("train_dataset/", one_hot=True)

# Parameters
learning_rate = 0.005
num_steps = 30000
batch_size = 64

# Network Parameters
image_dim = 784  # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
    'encoder_h2': tf.Variable(glorot_init([hidden_dim, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_h2': tf.Variable(glorot_init([hidden_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}

biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'encoder_b2': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder_b2': tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# Building the encoder
input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = tf.nn.tanh(tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1'])
# encoder = tf.nn.softplus(tf.matmul(encoder, weights['encoder_h2']) + biases['encoder_b2'])
# encoder = tf.nn.tanh(encoder)  # 激活函数：加入非线性因素，提高模型表达能力
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

# Sampler: Normal (gaussian) random distribution
# tf.random_normal中文文档：https://blog.csdn.net/dcrmg/article/details/79028043
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps  # 重参数技巧：z = mu + sigma*epsilon

# Building the decoder (with scope to re-use these layers later)
decoder = tf.nn.tanh(tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1'])
# decoder = tf.nn.softplus(tf.matmul(decoder, weights['decoder_h2']) + biases['decoder_b2'])
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)


# Define VAE Loss
# The loss is composed of two terms:
# 1.) The reconstruction loss (the negative log probability
#     of the input under the reconstructed Bernoulli distribution
#     induced by the decoder in the train_dataset space).
#     This can be interpreted as the number of "nats" required
#     for reconstructing the input when the activation in latent
#     is given.
#     Adding 1e-10 to avoid evaluation of log(0.0)
# 2.) The latent loss, which is defined as the Kullback Leibler divergence
#     between the distribution in latent space induced by the encoder on
#     the train_dataset and some prior. This acts as a kind of regularizer.
#     This can be interpreted as the number of "nats" required
#     for transmitting the the latent space distribution given
#     the prior.
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std) #tf.exp(z_std)=epsilon^2l
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op = vae_loss(decoder, input_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST train_dataset (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Train
        feed_dict = {input_image: batch_x}
        _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i, Loss: %f' % (i, l))

    # Testing
    # Generator takes noise as input
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
    # Rebuild the decoder to create image from noise
    decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

    # Building a manifold of generated digits
    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)

    canvas = np.empty((28 * n, 28 * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            # z_mu = np.array([[xi, yi, xi, yi,xi, yi,xi, yi,xi, yi]] * batch_size)
            z_mu = np.array([[xi, yi]] * batch_size)
            x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
            x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.show()