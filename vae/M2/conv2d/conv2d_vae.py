###
'''
Replication of M1 from http://arxiv.org/abs/1406.5298
Title: Semi-Supervised Learning with Deep Generative Models
Authors: Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling
Original Implementation (Theano): https://github.com/dpkingma/nips14-ssl
---
Code By: S. Saemundsson
---
Mod: encode层的全连接网络修改成了conv2d
'''

###
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import time

import sys

from ddi.data_process import reshape_to_4tensor

sys.path.extend(['/home/cdy/ykq/vae', '/home/cdy/ykq/vae/M2'])
import M2.utils as utils
from M2.conv2d.ConvolutionNetwork import ConvolutionallyConnected
from M2.neuralnetworks import FullyConnected
from prettytensor import bookkeeper


class Conv2dVariationalAutoencoder(object):
    def __init__(self,
                 dim_x, dim_z,
                 hidden_layers_px,
                 hidden_layers_qz,
                 batch_size = 100,
                 p_x='bernoulli',  # 离散取值
                 q_z='gaussian_marg',
                 p_z='gaussian_marg',
                 nonlin_px=tf.nn.softplus,
                 nonlin_qz=tf.nn.softplus,
                 l2_loss=0.1):

        self.dim_x, self.dim_z = dim_x, dim_z
        self.l2_loss = l2_loss
        self.batch_size = batch_size

        self.distributions = {'p_x': p_x,
                              'q_z': q_z,
                              'p_z': p_z}

        ''' Create Graph '''

        self.G = tf.Graph()

        with self.G.as_default():
            # self.x = tf.placeholder(tf.float32, [None, self.dim_x])
            self.x = tf.placeholder(tf.float32, [self.batch_size, self.dim_x])

            self.encoder = ConvolutionallyConnected(dim_output=2 * self.dim_z,
                                                    hidden_layers=hidden_layers_qz)

            self.decoder = ConvolutionallyConnected(dim_output=self.dim_x,
                                                    hidden_layers=hidden_layers_px)

            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session()

            # print('hidden_layers_px', hidden_layers_px)
            # print('hidden_layers_qz', hidden_layers_qz)

    def _draw_sample(self, mu, log_sigma_sq):

        epsilon = tf.random_normal((tf.shape(mu)), 0, 1)
        sample = tf.add(mu,
                        tf.multiply(
                            tf.exp(0.5 * log_sigma_sq), epsilon))

        return sample

    # 网络的输入
    def _generate_zx(self, x, phase=pt.Phase.train, reuse=False):
        # x = reshape_to_4tensor(x)
        with tf.variable_scope('encoder', reuse=reuse):
            encoder_out = self.encoder.output(x, phase=phase)
        z_mu, z_lsgms = encoder_out.split(split_dim=1, num_splits=2)
        z_sample = self._draw_sample(z_mu, z_lsgms)
        # z_sample = reshape_to_4tensor(z_sample)
        return z_sample, z_mu, z_lsgms

    def _generate_xz(self, z, phase=pt.Phase.train, reuse=False):
        # z = reshape_to_4tensor(z)
        with tf.variable_scope('decoder', reuse=reuse):
            x_recon_logits = self.decoder.output(z, phase=phase)
        x_recon = tf.nn.sigmoid(x_recon_logits)
        return x_recon, x_recon_logits

    def _objective(self):

        ############
        ''' Cost '''
        ############

        self.z_sample, self.z_mu, self.z_lsgms = self._generate_zx(self.x)
        # print("z_sample: ", self.z_sample.shape())
        # print("z_mu: ", self.z_mu.shape())
        # print("z_lsgms: ", self.z_lsgms.shape())
        self.x_recon, self.x_recon_logits = self._generate_xz(self.z_sample)

        if self.distributions['p_z'] == 'gaussian_marg':
            prior_z = tf.reduce_sum(utils.tf_gaussian_marg(self.z_mu, self.z_lsgms), 1)

        if self.distributions['q_z'] == 'gaussian_marg':
            post_z = tf.reduce_sum(utils.tf_gaussian_ent(self.z_lsgms), 1)

        if self.distributions['p_x'] == 'bernoulli':
            origin_x = tf.reshape(self.x, [self.batch_size, self.dim_x])
            self.log_lik = - tf.reduce_sum(utils.tf_binary_xentropy(origin_x, self.x_recon), 1)

        l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        # print("post_z: ", post_z.shape)
        # print("prior_z: ", prior_z.shape)
        # print("log_like: ", self.log_lik.shape)
        self.cost = tf.reduce_mean(post_z - prior_z - self.log_lik) + self.l2_loss * l2

        ##################
        ''' Evaluation '''
        ##################

        self.z_sample_eval, _, _ = self._generate_zx(self.x, phase=pt.Phase.test, reuse=True)
        self.x_recon_eval, _ = self._generate_xz(self.z_sample_eval, phase=pt.Phase.test, reuse=True)

        self.eval_log_lik = - tf.reduce_mean(tf.reduce_sum(utils.tf_binary_xentropy(self.x, self.x_recon_eval), 1))

    def train(self, x, x_valid,
              epochs, #num_batches,
              save_path=None,
              print_every=1,
              learning_rate=3e-4,
              beta1=0.9,
              beta2=0.999,
              seed=31415,
              stop_iter=100,
              load_path=None,
              draw_img=1):

        self.num_examples = x.shape[0]
        # self.num_batches = num_batches
        assert self.num_examples % self.batch_size == 0, '#Examples % #Batches != 0'

        # self.batch_size = self.num_examples // self.num_batches

        # x_size = x.shape[0]
        # x_valid_size = x.shape[0]
        # x = np.reshape(x, [x_size, self.dim_x, 1, 1])
        # x_valid = np.reshape(x_valid, [x_valid_size, self.dim_x, 1, 1])

        ''' Session and Summary '''
        if save_path is None:
            self.save_path = 'checkpoints/model_CONV2D-VAE_{}.cpkt'.format(
                time.strftime("%m-%d-%H%M%S", time.localtime()))
        else:
            self.save_path = save_path

        np.random.seed(seed)
        tf.set_random_seed(seed)

        with self.G.as_default():

            self.optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
            self.train_op = self.optimiser.minimize(self.cost)
            init = tf.global_variables_initializer()  # tf.initialize_all_variables()
            self._test_vars = None

        with self.session as sess:

            sess.run(init)

            # 实际上并没有执行
            if load_path == 'default':
                self.saver.restore(sess, self.save_path)
            elif load_path is not None:
                self.saver.restore(sess, load_path)

            training_cost = 0.
            best_eval_log_lik = - np.inf
            stop_counter = 0

            for epoch in range(epochs):

                ''' Shuffle Data '''
                np.random.shuffle(x)

                ''' Training '''

                for x_batch in utils.feed_numpy(self.batch_size, x):
                    training_result = sess.run([self.train_op, self.cost],
                                               feed_dict={self.x: x_batch})

                    training_cost = training_result[1]

                ''' Evaluation '''

                stop_counter += 1

                # 训练，更新不断跟新参数
                if epoch % print_every == 0:

                    test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
                    if test_vars:
                        if test_vars != self._test_vars:
                            self._test_vars = list(test_vars)
                            self._test_var_init_op = tf.initialize_variables(test_vars)
                        self._test_var_init_op.run()

                    # eval_log_lik, x_recon_eval = \
                    #     sess.run([self.eval_log_lik, self.x_recon_eval],
                    #              feed_dict={self.x: x_valid})
                    eval_log_lik = 0
                    x_recon_eval = 0
                    valid_times = x_valid.shape[0] / self.batch_size
                    for x_valid_batch in utils.feed_numpy(self.batch_size, x_valid):
                        log_lik, recon_eval = sess.run([self.eval_log_lik, self.x_recon_eval], feed_dict={self.x: x_valid_batch})
                        eval_log_lik += log_lik
                        x_recon_eval += recon_eval
                    eval_log_lik /= valid_times
                    x_recon_eval /= valid_times

                    if eval_log_lik > best_eval_log_lik:
                        best_eval_log_lik = eval_log_lik
                        self.saver.save(sess, self.save_path)
                        stop_counter = 0

                    utils.print_metrics(epoch + 1,
                                        ['Training', 'cost', training_cost],
                                        ['Validation', 'log-likelihood', eval_log_lik])

                ## 画图
                # if draw_img > 0 and epoch % draw_img == 0:
                #
                # 	import matplotlib
                # 	matplotlib.use('Agg')
                # 	import pylab
                # 	import seaborn as sns
                #
                # 	five_random = np.random.random_integers(x_valid.shape[0], size = 5)
                # 	x_sample = x_valid[five_random]
                # 	x_recon_sample = x_recon_eval[five_random]
                #
                # 	sns.set_style('white')
                # 	f, axes = pylab.subplots(5, 2, figsize=(8,12))
                # 	for i,row in enumerate(axes):
                #
                # 		row[0].imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
                # 		im = row[1].imshow(x_recon_sample[i].reshape(28, 28), vmin=0, vmax=1,
                # 			cmap=sns.light_palette((1.0, 0.4980, 0.0549), input="rgb", as_cmap=True))
                #
                # 		pylab.setp([a.get_xticklabels() for a in row], visible=False)
                # 		pylab.setp([a.get_yticklabels() for a in row], visible=False)
                #
                # 	f.subplots_adjust(left=0.0, right=0.9, bottom=0.0, top=1.0)
                # 	cbar_ax = f.add_axes([0.9, 0.1, 0.04, 0.8])
                # 	f.colorbar(im, cax=cbar_ax, use_gridspec=True)
                #
                # 	pylab.tight_layout()
                # 	pylab.savefig('img/recon-'+str(epoch)+'.png', format='png')
                # 	pylab.clf()
                # 	pylab.close('all')

                if stop_counter >= stop_iter:
                    print('Stopping VAE training')
                    print('No change in validation log-likelihood for {} iterations'.format(stop_iter))
                    print('Best validation log-likelihood: {}'.format(best_eval_log_lik))
                    print('Model saved in {}'.format(self.save_path))
                    break

    def encode(self, x, sample=False):

        if sample:
            return self.session.run([self.z_sample, self.z_mu, self.z_lsgms], feed_dict={self.x: x})
        else:
            return self.session.run([self.z_mu, self.z_lsgms], feed_dict={self.x: x})

    def decode(self, z):

        return self.session.run([self.x_recon],
                                feed_dict={self.z_sample: z})
