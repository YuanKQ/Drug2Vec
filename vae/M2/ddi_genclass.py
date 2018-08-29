###
'''
Replication of M2 from http://arxiv.org/abs/1406.5298
Title: Semi-Supervised Learning with Deep Generative Models
Authors: Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling
Original Implementation (Theano): https://github.com/dpkingma/nips14-ssl
---
Code By: S. Saemundsson
---
Mod: 在最终生成的regcon的X中添加了一个classifier
'''
###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np

import utils
import time

import sys
sys.path.append("/home/yuan/Code/PycharmProjects/vae/M2/")
from M2.neuralnetworks import FullyConnected
from prettytensor import bookkeeper

class DdiGenerativeClassifier( object ):

	def __init__(   self,
					dim_x, dim_z, dim_y,
					num_examples, num_lab, num_batches,
					p_x = 'gaussian',
					q_z = 'gaussian_marg',
					p_z = 'gaussian_marg',
					hidden_layers_px = [500],
					hidden_layers_qz = [500],
					hidden_layers_qy = [500],
					nonlin_px = tf.nn.softplus,
					nonlin_qz = tf.nn.softplus,
					nonlin_qy = tf.nn.softplus,
					alpha = 0.1,
					l2_loss = 0.0	):


		self.dim_x, self.dim_z, self.dim_y = int(dim_x), int(dim_z), int(dim_y)

		self.distributions = { 		  'p_x': 	p_x,
									  'q_z': 	q_z,
									  'p_z': 	p_z,
									  'p_y':	'uniform'	}

		self.num_examples = num_examples
		self.num_batches = num_batches
		self.num_lab = num_lab
		self.num_ulab = self.num_examples - num_lab

		assert self.num_lab % self.num_batches == 0, '#Labelled % #Batches != 0'
		assert self.num_ulab % self.num_batches == 0, '#Unlabelled % #Batches != 0'
		assert self.num_examples % self.num_batches == 0, '#Examples % #Batches != 0'

		self.batch_size = self.num_examples // self.num_batches
		self.num_lab_batch = self.num_lab // self.num_batches  # lab数据的batch_size
		self.num_ulab_batch = self.num_ulab // self.num_batches  # ulab数据的batch_size

		self.beta = alpha * ( float(self.batch_size) / self.num_lab_batch )

		''' Create Graph '''

		self.G = tf.Graph()

		with self.G.as_default():

			self.x_labelled_mu 			= tf.placeholder( tf.float32, [None, self.dim_x] )
			self.x_labelled_lsgms 		= tf.placeholder( tf.float32, [None, self.dim_x] )
			self.x_unlabelled_mu 		= tf.placeholder( tf.float32, [None, self.dim_x] )
			self.x_unlabelled_lsgms 	= tf.placeholder( tf.float32, [None, self.dim_x] )
			self.y_lab      			= tf.placeholder( tf.float32, [None, self.dim_y] )

			self.classifier = FullyConnected( 	dim_output 		= self.dim_y,
												 hidden_layers 	= hidden_layers_qy,
												 nonlinearity 	= nonlin_qy,
												 l2loss 			= l2_loss 	)

			self.encoder = FullyConnected( 		dim_output 		= 2 * self.dim_z,  # 隐变量空间的均值和方差
												  hidden_layers 	= hidden_layers_qz,
												  nonlinearity 	= nonlin_qz,
												  l2loss 			= l2_loss 	)

			self.decoder = FullyConnected( 		dim_output 		= 2 * self.dim_x,  #  mean_dim + var_dim = dim_x
												  hidden_layers 	= hidden_layers_px,
												  nonlinearity 	= nonlin_px,
												  l2loss 			= l2_loss 	)

			self._objective()
			self.saver = tf.train.Saver()
			self.session = tf.Session()

		# print('genclass: hidden_layers_px', hidden_layers_px)
		# print('genclass: hidden_layers_qz', hidden_layers_qz)
		# print('genclass: hidden_layers_qy', hidden_layers_qy)



	def _draw_sample( self, mu, log_sigma_sq ):
		# 这里就使用了重参数z = z_mean + tf.exp(z_std / 2) * eps  # 重参数技巧：z = mu + sigma*epsilon
		epsilon = tf.random_normal( ( tf.shape( mu ) ), 0, 1 )
		sample = tf.add( mu,
						 tf.multiply(
							 tf.exp( 0.5 * log_sigma_sq ), epsilon ) )

		return sample

	# 网络的输入
	def _generate_yx( self, x_mu, x_log_sigma_sq, phase = pt.Phase.train, reuse = False ):
		x_sample = self._draw_sample( x_mu, x_log_sigma_sq )
		with tf.variable_scope('classifier', reuse = reuse):  # 变量共享，问题在于共享哪一个变量呢？
			y_logits = self.classifier.output( x_sample, phase )

		return y_logits, x_sample

	def _generate_zxy( self, x, y, reuse = False ):

		with tf.variable_scope('encoder', reuse = reuse):
			encoder_out = self.encoder.output( tf.concat( [x, y], 1 ) )
		z_mu, z_lsgms   = encoder_out.split( split_dim = 1, num_splits = 2 )
		z_sample        = self._draw_sample( z_mu, z_lsgms )

		return z_sample, z_mu, z_lsgms

	def _generate_xzy( self, z, y, reuse = False ):

		with tf.variable_scope('decoder', reuse = reuse):
			decoder_out = self.decoder.output( tf.concat( [z, y] , 1) )
		x_recon_mu, x_recon_lsgms   = decoder_out.split( split_dim = 1, num_splits = 2 )

		return x_recon_mu, x_recon_lsgms

	def _objective( self ):

		###############
		''' L(x,y) '''
		###############

		def L(x_recon, x, y, z):

			if self.distributions['p_z'] == 'gaussian_marg':

				log_prior_z = tf.reduce_sum( utils.tf_gaussian_marg( z[1], z[2] ), 1 )

			elif self.distributions['p_z'] == 'gaussian':

				log_prior_z = tf.reduce_sum( utils.tf_stdnormal_logpdf( z[0] ), 1 )

			if self.distributions['p_y'] == 'uniform':

				y_prior = (1. / self.dim_y) * tf.ones_like( y )
				log_prior_y = - tf.nn.softmax_cross_entropy_with_logits( labels=y_prior, logits=y )

			if self.distributions['p_x'] == 'gaussian':

				log_lik = tf.reduce_sum( utils.tf_normal_logpdf( x, x_recon[0], x_recon[1] ), 1 )

			if self.distributions['q_z'] == 'gaussian_marg':

				log_post_z = tf.reduce_sum( utils.tf_gaussian_ent( z[2] ), 1 )

			elif self.distributions['q_z'] == 'gaussian':

				log_post_z = tf.reduce_sum( utils.tf_normal_logpdf( z[0], z[1], z[2] ), 1 )

			_L = log_prior_y + log_lik + log_prior_z - log_post_z

			return  _L

		def glorot_init(shape):
			# return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
			return tf.constant(1.0, shape=shape)

		self.weights = {
			'lab_clf': tf.Variable(glorot_init([self.num_lab_batch, 1])),
			'lab_recon_clf': tf.Variable(glorot_init([self.num_lab_batch, 1])),
			'ulab_clf': tf.Variable(glorot_init([self.num_ulab_batch, 1])),
			'ulab_recon_clf': tf.Variable(glorot_init([self.num_ulab_batch, 1])),
			'L_loss': tf.Variable(glorot_init([self.num_lab_batch, 1])),
			'U_loss': tf.Variable(glorot_init([self.num_lab_batch, 1])),
			'loss_weights': tf.Variable(glorot_init([1, 2])),
		}

		###########################
		''' Labelled Datapoints '''
		###########################
		# self._y_lab_logits, self.x_lab = self._generate_yx(self.x_labelled_mu, self.x_labelled_lsgms)
		self.y_lab_logits, self.x_lab = self._generate_yx(self.x_labelled_mu, self.x_labelled_lsgms)
		self.z_lab, self.z_lab_mu, self.z_lab_lsgms = self._generate_zxy( self.x_lab, self.y_lab )
		self.x_recon_lab_mu, self.x_recon_lab_lsgms = self._generate_xzy( self.z_lab, self.y_lab )
		# self.y_lab_logits, _ = self._generate_yx(self.x_recon_lab_mu, self.x_recon_lab_lsgms, reuse = True)

		L_lab = L(  [self.x_recon_lab_mu, self.x_recon_lab_lsgms], self.x_lab, self.y_lab,
					[self.z_lab, self.z_lab_mu, self.z_lab_lsgms] )

		# 交叉熵计算loss代价： https://blog.csdn.net/mao_xiao_feng/article/details/53382790
		# L_lab += - self.beta * (tf.multiply(self.weights['lab_clf'], tf.nn.softmax_cross_entropy_with_logits( labels=self.y_lab_logits, logits=self.y_lab )) +
		# 						tf.multiply(self.weights['lab_recon_clf'], tf.nn.softmax_cross_entropy_with_logits( labels=self._y_lab_logits, logits=self.y_lab )))/2
		L_lab += - self.beta * (tf.nn.softmax_cross_entropy_with_logits(labels=self.y_lab_logits, logits=self.y_lab))

		############################
		''' Unabelled Datapoints '''
		############################

		def one_label_tensor( label, num_ulab_batch ):

			indices = []
			values = []
			for i in range(num_ulab_batch):
				indices += [[ i, label ]]
				values += [ 1. ]

			_y_ulab = tf.sparse_tensor_to_dense(
				tf.SparseTensor( indices=indices, values=values, dense_shape=[ num_ulab_batch, self.dim_y ] ), 0.0 )

			return _y_ulab

		self.y_ulab_logits, self.x_ulab = self._generate_yx( self.x_unlabelled_mu, self.x_unlabelled_lsgms, reuse = True )
		# self._y_ulab_logits_, self.x_ulab = self._generate_yx( self.x_unlabelled_mu, self.x_unlabelled_lsgms, reuse = True )

		for label in range(self.dim_y):

			_y_ulab = one_label_tensor( label, self.num_ulab_batch )  # 随机产生的
			self.z_ulab, self.z_ulab_mu, self.z_ulab_lsgms = self._generate_zxy( self.x_ulab, _y_ulab, reuse = True )
			self.x_recon_ulab_mu, self.x_recon_ulab_lsgms = self._generate_xzy( self.z_ulab, _y_ulab, reuse = True )

			_L_ulab =   tf.expand_dims(
				L(  [self.x_recon_ulab_mu, self.x_recon_ulab_lsgms], self.x_ulab, _y_ulab,
					[self.z_ulab, self.z_ulab_mu, self.z_ulab_lsgms]), 1)

			if label == 0: L_ulab = tf.identity( _L_ulab )
			else: L_ulab = tf.concat( [L_ulab, _L_ulab], 1 )

		# self.y_ulab_logits, _ = self._generate_yx(self.x_recon_ulab_mu, self.x_recon_ulab_lsgms, reuse = True)
		# self.y_ulab = (tf.multiply(self.weights['ulab_clf'], self.y_ulab_logits.softmax_activation()) + tf.multiply(self.weights['ulab_recon_clf'] ,self._y_ulab_logits_.softmax_activation()))/2
		self.y_ulab = self.y_ulab_logits.softmax_activation()

		U = tf.reduce_sum(
			tf.multiply( self.y_ulab,
						 tf.subtract( L_ulab,
									  tf.log( self.y_ulab ) ) ), 1 )

		########################
		''' Prior on Weights '''
		########################

		L_weights = 0.
		_weights = tf.trainable_variables()
		for w in _weights:
			L_weights += tf.reduce_sum( utils.tf_stdnormal_logpdf( w ) )

		##################
		''' Total Cost '''
		##################
		# self.weights['L_loss'] = tf.nn.softmax(self.weights['L_loss'])
		# self.weights['U_loss'] = tf.nn.softmax(self.weights['U_loss'])
		# tmp = tf.add(self.weights['L_loss'], self.weights['U_loss'])
		# self.weights['L_loss'] = tf.divide(self.weights['L_loss'], tmp)
		# self.weights['U_loss'] = tf.divide(self.weights['U_loss'], tmp)
		#L_lab_tot = tf.reduce_sum( tf.multiply(self.weights['L_loss'], L_lab ))
		#U_tot = tf.reduce_sum( tf.multiply(self.weights['U_loss'], U ))
		self.weights['loss_weights'] = tf.nn.softmax(self.weights['loss_weights'])
		L_lab_tot = tf.reduce_sum(L_lab)
		U_tot = tf.reduce_sum(U)
		tot = [L_lab_tot, U_tot]

		# self.cost = ( ( tf.reduce_sum(tf.multiply(tot, self.weights['loss_weights']) )) * self.num_batches + L_weights) / (
		# 		- self.num_batches * self.batch_size )
		self.cost = ( ( L_lab_tot + U_tot ) * self.num_batches + L_weights) / (
				- self.num_batches * self.batch_size )


		##################
		''' Evaluation '''
		##################

		self.y_test_logits, _ = self._generate_yx(self.x_labelled_mu, self.x_labelled_lsgms,
						 						  phase=pt.Phase.test, reuse=True)
		self.y_test_pred = self.y_test_logits.softmax(self.y_lab)

		# _y_test_logits, x_test_lab = self._generate_yx(self.x_labelled_mu, self.x_labelled_lsgms,
		# 											   phase=pt.Phase.test, reuse=True)
        #
		# # z_test_lab, z_test_lab_mu, z_test_lab_lsgms = self._generate_zxy(x_test_lab, _y_test_logits, reuse=True)
		# # x_test_recon_mu, x_test_recon_lsgms = self._generate_xzy(z_test_lab, _y_test_logits, reuse=True)
        #
		# for label in range(self.dim_y):
		# 	_y_ulab = one_label_tensor(label, 600)  # 随机产生的
		# 	z_test_lab, z_test_lab_mu, z_test_lab_lsgms = self._generate_zxy(x_test_lab, _y_ulab, reuse=True)
		# 	x_test_recon_mu, x_test_recon_lsgms = self._generate_xzy(z_test_lab, _y_ulab, reuse=True)
        #
		# 	self.y_test_logits, _ = self._generate_yx(x_test_recon_mu, x_test_recon_lsgms, reuse=True)
		# 	## self.y_ulab = (tf.multiply(self.weights['ulab_clf'], self.y_ulab_logits.softmax_activation()) + tf.multiply(self.weights['ulab_recon_clf'] ,self._y_ulab_logits_.softmax_activation()))/2
        #
		# 	if label == 0: self.y_test_pred = self.y_test_logits
		# 	else: self.y_test_pred += self.y_test_logits
		# self.y_test_pred = self.y_test_pred.softmax(self.y_lab)

		# self.eval_accuracy = self.y_test_pred \
		# 	.softmax.evaluate_classifier( self.y_lab, phase = pt.Phase.test )
		# self.eval_cross_entropy = self.y_test_pred.loss
		# self.eval_precision, self.eval_recall = self.y_test_pred.softmax \
		# 	.evaluate_precision_recall( self.y_lab, phase = pt.Phase.test )
		self.eval_accuracy = self.y_test_pred.softmax.evaluate_classifier(self.y_lab)
		self.eval_cross_entropy = self.y_test_pred.loss
		self.eval_precision, self.eval_recall = self.y_test_pred.softmax.evaluate_precision_recall( self.y_lab)



	def train(      self, x_labelled, y, x_unlabelled,
					epochs,
					x_valid, y_valid,
					print_every = 1,
					learning_rate = 3e-4,
					beta1 = 0.9,
					beta2 = 0.999,
					seed = 31415,
					stop_iter = 100,
					save_path = None,
					load_path = None    ):


		''' Session and Summary '''
		if save_path is None:
			self.save_path = 'checkpoints/model_GC_{}-{}-{}_{}.cpkt'.format(
				self.num_lab,learning_rate,self.batch_size,time.time())
		else:
			self.save_path = save_path

		np.random.seed(seed)
		tf.set_random_seed(seed)

		with self.G.as_default():

			self.optimiser = tf.train.AdamOptimizer( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 )
			self.train_op = self.optimiser.minimize( self.cost )
			init = tf.initialize_all_variables()
			self._test_vars = None


		_data_labelled = np.hstack( [x_labelled, y] )
		_data_unlabelled = x_unlabelled
		x_valid_mu, x_valid_lsgms = x_valid[ :, :int(self.dim_x) ], x_valid[ :, int(self.dim_x): int(2*self.dim_x) ]

		with self.session as sess:

			sess.run(init)
			if load_path == 'default': self.saver.restore( sess, self.save_path )
			elif load_path is not None: self.saver.restore( sess, load_path )

			best_eval_accuracy = 0.
			best_train_accuracy = 0.
			stop_counter = 0

			print("****lab_clf", self.weights['lab_clf'])
			print("****lab_recon_clf", self.weights['lab_recon_clf'])
			print("****ulab_clf", self.weights['ulab_clf'])
			print("****ulab_recon_clf", self.weights['ulab_recon_clf'])
			print("****L_loss", self.weights['L_loss'])
			print("****U_loss", self.weights['U_loss'])

			for epoch in range(epochs):

				''' Shuffle Data '''
				np.random.shuffle( _data_labelled )
				np.random.shuffle( _data_unlabelled )

				''' Training '''

				for x_l_mu, x_l_lsgms, y, x_u_mu, x_u_lsgms in utils.feed_numpy_semisupervised(
						self.num_lab_batch, self.num_ulab_batch,
						_data_labelled[:,:2*self.dim_x], _data_labelled[:,2*self.dim_x:],_data_unlabelled ):

					training_result = sess.run( [self.train_op, self.cost],
												feed_dict = {	self.x_labelled_mu:			x_l_mu,
																 self.x_labelled_lsgms: 		x_l_lsgms,
																 self.y_lab: 				y,
																 self.x_unlabelled_mu: 		x_u_mu,
																 self.x_unlabelled_lsgms: 	x_u_lsgms} )

					training_cost = training_result[1]


				''' Evaluation '''

				stop_counter += 1

				if epoch % print_every == 0:

					test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
					if test_vars:
						if test_vars != self._test_vars:
							self._test_vars = list(test_vars)
							self._test_var_init_op = tf.initialize_variables(test_vars)
						self._test_var_init_op.run()


					eval_accuracy, eval_cross_entropy = \
						sess.run( [self.eval_accuracy, self.eval_cross_entropy],
								  feed_dict = { 	self.x_labelled_mu: 	x_valid_mu,
												   self.x_labelled_lsgms:	x_valid_lsgms,
												   self.y_lab:				y_valid } )
					if eval_accuracy > best_eval_accuracy:

						best_eval_accuracy = eval_accuracy
						self.saver.save( sess, self.save_path )
						stop_counter = 0
						print("accuracy update: ", best_eval_accuracy)

					utils.print_metrics( 	epoch+1,
											['Training', 'cost', training_cost],
											['Validation', 'accuracy', eval_accuracy],
											['Validation', 'cross-entropy', eval_cross_entropy] )



				if stop_counter >= stop_iter:
					print('Stopping GC training')
					print('No change in validation accuracy for {} iterations'.format(stop_iter))
					print('Best validation accuracy: {}'.format(best_eval_accuracy))
					print('Model saved in {}'.format(self.save_path))
					break

	def predict_labels( self, x_test, y_test ):

		test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
		tf.initialize_variables(test_vars).run()

		x_test_mu = x_test[:,:self.dim_x]
		x_test_lsgms = x_test[:,self.dim_x:2*self.dim_x]

		accuracy, cross_entropy, precision, recall = \
			self.session.run( [self.eval_accuracy, self.eval_cross_entropy, self.eval_precision, self.eval_recall],
							  feed_dict = {self.x_labelled_mu: x_test_mu, self.x_labelled_lsgms: x_test_lsgms, self.y_lab: y_test} )

		utils.print_metrics(	'X',
								['Test', 'accuracy', accuracy],
								['Test', 'cross-entropy', cross_entropy],
								['Test', 'precision', precision],
								['Test', 'recall', recall] )
