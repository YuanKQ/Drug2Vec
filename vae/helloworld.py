# -*- coding: utf-8 -*-
"""
__title__ = 'helloworld.py'
__IDE__ = 'PyCharm'
__author__ = 'YuanKQ'
__mtime__ = 'May 17,2018 10:19'
__mail__ = kq_yuan@outlook.com

__description__==

"""
import tensorflow as tf
a = 2
b = 3
d =[[a, b], [10, 20]]

result = tf.reduce_sum(d)
tf.initialize_all_variables()
with tf.Session() as sess:
    print(sess.run(result))
    print(tf.shape(result))
    print(result.get_shape())
    print(result.get_shape()[0])
    print(result.get_shape()[1])


