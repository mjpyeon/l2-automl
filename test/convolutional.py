from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

BATCH_SIZE = 100
INPUT_DIM = 50
OUTPUT_DIM = 1
DIM1 = 25

EVAL_FREQUENCY = 100  # Number of steps between evaluations.


FLAGS = None


X = np.random.rand(INPUT_DIM, 100)
W = np.random.rand(INPUT_DIM, OUTPUT_DIM)
noise = np.random.normal(size = 100)
Y = np.matmul(np.transpose(X), W).squeeze(1) + noise
Y = np.resize(Y, (BATCH_SIZE, OUTPUT_DIM))
print(Y.shape)

train_X_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, INPUT_DIM))
train_Y_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_DIM))

fc1_weights = tf.get_variable("fc1_weights", [INPUT_DIM, DIM1], dtype=tf.float32,
  initializer=tf.zeros_initializer)
fc2_weights = tf.get_variable("fc2_weights", [DIM1, OUTPUT_DIM], dtype=tf.float32,
  initializer=tf.zeros_initializer)

#output1 = train_X_input * fc1_weights
output1 = tf.matmul(train_X_input, fc1_weights)
output2 = tf.matmul(output1, fc2_weights)

loss = tf.losses.mean_squared_error(output2, train_Y_input)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
  writer = tf.summary.FileWriter('/tmp/train', sess.graph)
  tf.global_variables_initializer().run()
  print('Initialized!')
  # Loop through training steps.
  for step in xrange(100):
    feed_dict = {train_X_input: np.transpose(X),
                 train_Y_input: Y}
    # Run the optimizer to update weights.
    sess.run(optimizer, feed_dict=feed_dict)
