# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: dim_convert.py
@time: 18-6-20 下午10:38
@description:
"""
import tensorflow as tf
import numpy as np
ll = [[10, 11,12,13, 14], [20, 21,22,23,24], [30, 31, 32, 33, 34]]
lll = [[11, 22,33,44, 55], [55, 66,77, 88, 99], [99, 111, 222, 333, 444]]
# ll = tf.reshape(ll, [2,4,1,1])
# lll = tf.reshape(lll, [2,4,1,1])
print("ll:", np.shape(ll))
c = tf.stack([ll, lll], axis=0)
cc = tf.stack([ll, lll], axis=1)
ccc = tf.stack([ll, lll], axis=2)
# cccc = tf.stack([ll, lll], axis=3)
with tf.Session() as sess:
    sess.run(c)
    sess.run(cc)
    sess.run(ccc)
    # sess.run(cccc)
    print("c: ", c.shape)
    print("cc: ", cc.shape)
    print("ccc: ", ccc.shape)
    print("c: ", sess.run(c))
    print("cc: ",sess.run(cc))
    print("ccc: ",sess.run(ccc))
    # print(sess.run(tf.reshape(ccc,[3,4,1,4])))
    # print("cccc: ", cccc.shape)
