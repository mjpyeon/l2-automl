# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math
import tensorflow as tf
SEED = 1
tf.set_random_seed(SEED)
import cifar10
import pdb
import os
import numpy as np
import pdb
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train_optimizer',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 300,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('val_num_examples', 10000,
                            """Number of examples to run.""")
num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_epoch(optimizer_code, lr=0.1, Full_train=False, log=False):
  cifar10.INITIAL_LEARNING_RATE = lr
  best_loss = 1e10
  last_ten_loss = [0] * 10
  tf.reset_default_graph()
  if Full_train:
    last_step = 5 * num_batches_per_epoch
  else:
    last_step = num_batches_per_epoch
  with tf.Graph().as_default():
    tf.set_random_seed(SEED)
    global_step = tf.train.get_or_create_global_step()
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()
    logits = cifar10.inference(images, is_train=True)
    loss = cifar10.loss(logits, labels)
    tf_optimizer_code = tf.placeholder(tf.int32)
    train_op = cifar10.train(loss, global_step, tf_optimizer_code)
    #saver = tf.train.Saver()
    val_images, val_labels = cifar10.inputs(eval_data=False)
    val_logits = cifar10.inference(val_images, is_train=False)
    val_top_k_op = tf.nn.in_top_k(val_logits, val_labels, 1)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
    with tf.train.MonitoredTrainingSession(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      out_logits, out_loss = sess.run([logits, loss])      
      pdb.set_trace()
      pass
    '''
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step != 0 and self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          if log:
            print (format_str % (datetime.now(), self._step, loss_value,
                                 examples_per_sec, sec_per_batch))
    '''
    with tf.train.MonitoredTrainingSession(
	     hooks=[tf.train.StopAtStepHook(last_step=last_step),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
    '''
    def eval_once(sess):
      val_num_iter = int(math.ceil(FLAGS.val_num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = val_num_iter * FLAGS.batch_size
      for step in range(val_num_iter):
        predictions = sess.run([val_top_k_op])
        true_count += np.sum(predictions)
      precision = float(true_count) / total_sample_count
      return precision

    with tf.Session() as mon_sess:
      mon_sess.run(tf.global_variables_initializer())
      coord=tf.train.Coordinator()
      threads = tf.train.start_queue_runners(mon_sess, coord)
      tf.train.start_queue_runners(mon_sess)
      global_step_out = 0
      pdb.set_trace()
      while global_step_out < last_step:
        loss_out, _ = mon_sess.run([loss, train_op], {tf_optimizer_code:optimizer_code})
        if (math.isnan(loss_out) or loss_out > 1e2):
          pdb.set_trace()
          return 0
        last_ten_loss.pop(0)
        last_ten_loss.append(loss_out)
        global_step_out = mon_sess.run(global_step)
        if global_step_out % num_batches_per_epoch == 0:
          print('step {}: valid acc {}'.format(global_step_out, eval_once(mon_sess)))
        #if(best_loss > loss_out):
        #  best_loss = loss_out
      #saver.save(mon_sess, FLAGS.train_dir+'/model.ckpt')
      precision = eval_once(mon_sess)
      coord.request_stop()
      coord.join(threads) 
  #with tf.Graph().as_default():

  return precision #sum(last_ten_loss) / len(last_ten_loss)



def main(optimizer_code=None):  # pylint: disable=unused-argument
  #optimizer_code = [14, 10, 8, 11, 5, 11, 17, 2, 1, 3, 15, 18, 4, 2, 0, 17, 18, 8, 1, 2] # this gives 100% acc on valid set after 4 epochs of training
  optimizer_code = [ 0,12, 0, 7, 0, 3,17, 7,11, 4,12,18, 0, 1, 4,12,17,11, 0, 2]
  print(optimizer_code)
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    #assert False
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  best_lr = 1e-1
  best_acc = 0
  '''
  for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]:
  #for lr in [1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e0]:  
    #print("learning rate: ", lr)
    acc = train_epoch(optimizer_code, lr, log=False)
    if acc > best_acc:
      best_acc = acc
      best_lr = lr
  '''
  cifar10.INITIAL_LEARNING_RATE = best_lr
  #if best_loss > 4.6:
  #  return best_loss
  #else:
  print('best lr found {}, 1 epoch acc: {}'.format(best_lr, best_acc),)
  acc = train_epoch(optimizer_code, best_lr, True)
  print(', 5 epochs acc {}'.format(acc))
  return acc


if __name__ == '__main__':
  existing_codes = {'adam':[15,0,0,0,5],'rmsprop':[16,0,0,0,5],'momentum':[3,0,0,0,5],'sgd':[0,0,0,0,5]}
  for k,v in existing_codes.iteritems():
    print(k, main(optimizer_code=v))
  #tf.app.run()
