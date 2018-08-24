# encoding: utf-8

"""
@author: Kaiqi Yuan
@software: PyCharm
@file: data_view.py
@time: 18-7-2 下午4:27
@description:
"""
import sys
import os
import csv
import prettytensor as pt
import tensorflow as tf
import numpy as np
sys.path.extend(['/home/cdy/ykq/DDISuccess', '/home/cdy/ykq/DDISuccess/LstmDemo'])
from LstmDemo import data_utils
# import tensorflow as tf


BATCH_SIZE = 32
TIMESTEPS = 15

names, sex, lengths = data_utils.baby_names(TIMESTEPS)
print("names: ", names.shape)

def baby_names(max_length=15):
  """Opens the baby_names csv file and produces numpy array.

  Args:
    max_length: The maximum length, 15 was the longest name when this was
      written.  Short entries will be padded with the EOS marker.
  Returns:
    A numpy array of the names converted to ascii codes, the labels and an
    array of lengths.
  Raises:
    ValueError: if max_length is too small.
  """
names = []
lengths = []
targets = []
UNK = 0
EOS = 1

def convert_to_int(char):
  i = ord(char)
  if i >= 128:
      return UNK
  return i

def baby_names(max_length=15):
  with open(os.path.join(os.path.dirname(sys.modules[__name__].__file__),
                         'baby_names.csv'), 'rt', encoding="utf-8") as f:
    first = True
    for l in csv.reader(f, delimiter=','):
      if first:
        first = False
        continue
      assert len(l) == 4, l
      name = l[0]
      if max_length < len(name):
        raise ValueError('Max length is too small: %d > %d' %
                         (max_length, len(name)))
      chars = [convert_to_int(c) for c in name]
      names.append(chars + ([EOS] * (max_length - len(chars))))
      lengths.append([len(name)])
      values = [float(l[2]), float(l[3])]
      if abs(sum(values) - 1) > 0.001:
        raise ValueError('Each row must sum to 1: %s' % l)
      targets.append(values)

  return np.array(names), np.array(targets), np.array(lengths)

def permute_data(arrays, random_state=None):
  """Permute multiple numpy arrays with the same order."""
  if any(len(a) != len(arrays[0]) for a in arrays):
    raise ValueError('All arrays must be the same length.')
  if not random_state:
    random_state = np.random
  order = random_state.permutation(len(arrays[0]))
  return [a[order] for a in arrays]


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


if __name__ == '__main__':
    l = np.array([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 13, 24]])
    # l = np.array([[[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 13, 24]], [[5, 6, 7, 8], [15, 16, 17, 18], [25, 26, 26, 28]]])
    print(l.shape)
    dims = [1, 0]
    ll = pt.wrap(tf.transpose(l, dims))
    lll = pt.wrap(tf.transpose(l, dims)).reshape([-1, 1])
    with tf.Session() as sess:
        print(sess.run(ll))
        print(sess.run(lll))
    """
    [[ 1 11 21]
     [ 2 12 22]
     [ 3 13 13]
     [ 4 14 24]]
    [[ 1]
     [11]
     [21]
     [ 2]
     [12]
     [22]
     [ 3]
     [13]
     [13]
     [ 4]
     [14]
     [24]]
    """

    l = np.arange(24).reshape([2, 3, 4])