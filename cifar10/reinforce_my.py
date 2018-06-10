from __future__ import print_function
from collections import deque

from my_reinforce import PolicyGradientREINFORCE
import tensorflow as tf
import numpy as np
import bisect
import pdb
import os
import cifar10_train1 as verifier
tf.set_random_seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format('my_reinforce'))

n_hidden_units = 150
num_actions = [17,12,6]
embedding_size = 100
MAX_LENGTH    = 5
MAX_EPISODES = 10000
saver_path_base = 'model/'
saver_path = None#'model/0'

class lstm_policy_network:
  def __init__(self):
    # 0:<s>, 1-17:operand_idx, 18-29:unary_idx, 30-35:binary_idx
    self.dsl_embeddings = tf.get_variable("dsl_embeddings", [sum(num_actions) + 1, embedding_size])
    self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    self.Ws = []
    self.Bs = []
    for i in range(3):
      self.Ws.append(tf.get_variable("W_lstm_"+str(i), [n_hidden_units, num_actions[i]],
                           initializer=tf.random_normal_initializer(stddev=0.1)))
      self.Bs.append(tf.get_variable("b_lstm_"+str(i), [num_actions[i]],
                           initializer=tf.constant_initializer(0)))
  def _run_rnn(self, emb, init_state):
    return tf.nn.dynamic_rnn(self.lstm_cell, emb, initial_state=init_state, time_major=False)

  def multiple_step(self, input):
    input = tf.expand_dims(input, 0) # [1, slen, input_dim], assume batchSize == 1
    emb = tf.squeeze(tf.nn.embedding_lookup(self.dsl_embeddings, input), 2) # [1, slen, embedding_size]
    init_state = self.lstm_cell.zero_state(1, dtype=tf.float32)
    outputs, final_state = self._run_rnn(emb, init_state)
    for idx in range(MAX_LENGTH):
      #output_logit = output_logit.assign(tf.zeros([1 ,sum(num_actions) + 1]))
      #output_logit = tf.Variable(tf.zeros([1 ,sum(num_actions) + 1]), name="output_logit")
      block_idx = bisect.bisect_left([1,3,4], idx % 5)
      final_output = tf.matmul(outputs[:,idx], self.Ws[block_idx]) + self.Bs[block_idx]
      input_start = 1 + sum(num_actions[:block_idx])
      input_end = input_start + num_actions[block_idx]
      if(idx == 0):
        output_logits = tf.concat([tf.zeros([1,input_start]), final_output, tf.zeros([1, sum(num_actions) + 1 - input_end])], 1)
        #output_logits = output_logit[:, input_start:input_end].assign(final_output)
      else:
        #output_logits = tf.concat([output_logits, output_logit[:, input_start:input_end].assign(final_output)], 0)
        output_new = tf.concat([tf.zeros([1,input_start]), final_output, tf.zeros([1, sum(num_actions) + 1 - input_end])], 1)
        output_logits = tf.concat([output_logits, output_new], 0)
      #print("output_logits:", output_logits)
    print(output_logits)
    return output_logits

  def init_state(self):
    return self.lstm_cell.zero_state(1, dtype=tf.float32)

  def step(self, input, idx, block_idx, hidden):
    #idx = tf.Print(idx, [idx, block_idx], 'step idx/block_idx:', summarize=20)
    input = tf.expand_dims(input, 0) # [1, slen, input_dim], assume batchSize == 1
    emb = tf.squeeze(tf.nn.embedding_lookup(self.dsl_embeddings, input), 2) # [1, slen, embedding_size]
    init_state = tf.cond(tf.equal(idx, 0)[0], lambda: self.lstm_cell.zero_state(1, dtype=tf.float32), lambda: hidden)
    #init_state = hidden
    outputs, final_state = self._run_rnn(emb, init_state)
    #block_idx = bisect.bisect_left([1,3,4], idx % 5)
    def _step_linear(_block_idx):
      return tf.matmul(outputs[:,-1], self.Ws[_block_idx]) + self.Bs[_block_idx]
    final_output = tf.cond(tf.less(block_idx, 1)[0], lambda:_step_linear(0), lambda:
                                                  tf.cond(tf.less(block_idx, 2)[0], lambda: _step_linear(1),
                                                          lambda: _step_linear(2)))
    #final_output = tf.Print(final_output, [tf.shape(final_output)], "step final output shape:", summarize=20)
    #final_output = tf.matmul(outputs[-1], tf.gather(self.Ws, block_idx)) + tf.gather(self.Bs, block_idx)
    return final_output, final_state



  '''
  def __call__(self, input, idx, hidden=None, multiple=False):
    input = tf.expand_dims(input, 0) # [1, slen, input_dim], assume batchSize == 1
    emb = tf.squeeze(tf.nn.embedding_lookup(self.dsl_embeddings, input), 2) # [1, slen, embedding_size]
    #print(emb)
    if idx != 0:
      init_state = hidden
    else:
      init_state = self.lstm_cell.zero_state(1, dtype=tf.float32)
    #print(init_state, emb)
    outputs, final_state = tf.nn.dynamic_rnn(self.lstm_cell, emb, initial_state=init_state, time_major=False)
    if not multiple:
      # map 0,1 to 0, 2,3 to 1, 4 to 2.
      block_idx = bisect.bisect_left([1,3,4], idx % 5)
      final_output = tf.matmul(outputs[-1], self.Ws[block_idx]) + self.Bs[block_idx]
      return final_output, final_state   
    else:
      #output_logits = tf.Variable(tf.zeros([MAX_LENGTH ,sum(num_actions) + 1]), name="output_logits")
      #output_logit = tf.Variable(tf.zeros([1 ,sum(num_actions) + 1]), name="output_logit")
      for idx in range(MAX_LENGTH):
        #output_logit = output_logit.assign(tf.zeros([1 ,sum(num_actions) + 1]))
        #output_logit = tf.Variable(tf.zeros([1 ,sum(num_actions) + 1]), name="output_logit")
        block_idx = bisect.bisect_left([1,3,4], idx % 5)
        final_output = tf.matmul(outputs[:,idx], self.Ws[block_idx]) + self.Bs[block_idx]
        input_start = 1 + sum(num_actions[:block_idx])
        input_end = input_start + num_actions[block_idx]
        if(idx == 0):
          output_logits = tf.concat([tf.zeros([1,input_start]), final_output, tf.zeros([1, sum(num_actions) + 1 - input_end])], 1)
          #output_logits = output_logit[:, input_start:input_end].assign(final_output)
        else:
          #output_logits = tf.concat([output_logits, output_logit[:, input_start:input_end].assign(final_output)], 0)
          output_new = tf.concat([tf.zeros([1,input_start]), final_output, tf.zeros([1, sum(num_actions) + 1 - input_end])], 1)
          output_logits = tf.concat([output_logits, output_new], 0)
        #print("output_logits:", output_logits)
      print(output_logits)
      return output_logits
  '''

pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       lstm_policy_network(),
                                       num_actions,
                                       n_hidden_units,
                                       saver_path=saver_path,
                                       summary_writer=writer)

'''
'''
best_code = []
best_reward = -100
episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):
  # initialize
  total_rewards = 0
  last_action = 0
  last_hidden = None
  code = []
  input = 0
  #pdb.set_trace()
  for t in range(MAX_LENGTH):
    block_idx = bisect.bisect_left([1,3,4], t % 5)
    action = pg_reinforce.sampleAction(np.array([[input]]), t)
    #action = pg_reinforce.randomSampleAction(t)
    code += [action]
    last_action = action + 1 + sum(num_actions[:block_idx])
    if t != MAX_LENGTH - 1:
      pg_reinforce.storeRollout(input, last_action, 0)
      pass
    else:
      reward = sum(code)
      print(code)
      reward =  -(verifier.main(None, code))
      print("reward:", reward)
      # if nan loss
      if reward == -100:
        reward = -5
      # if loss is around same with initial loss
      elif reward < -4.5:
        reward = -4.5
      pg_reinforce.storeRollout(input, last_action, reward)
    input = last_action
  pg_reinforce.updateModel()
  episode_history.append(reward)
  mean_rewards = np.mean(episode_history)
  if reward > best_reward:
        best_code = code
        best_reward = reward
  #if i_episode % 10 == 0:
  #  pg_reinforce.save_model(saver_path_base+str(i_episode))
  if i_episode % 10 == 0:
    print("Best code {} / rewards {}".format(', '.join([str(c) for c in best_code]), best_reward))
  print("Episode {}".format(i_episode))
  print("code {}".format(', '.join([str(c) for c in code])))
  print("Reward for this episode: {}".format(reward))
  print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
  