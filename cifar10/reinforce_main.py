from __future__ import print_function
from collections import deque

from reinforce_module import PolicyGradientREINFORCE
import tensorflow as tf
import numpy as np
import bisect
import pdb
import os
import cifar10_train as verifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session()
tf.set_random_seed(1)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.summary.FileWriter("/tmp/{}-experiment-1".format('my_reinforce'))

n_hidden_units = 150
num_actions = [17,12,6]
embedding_size = 100
MAX_LENGTH    = 5
MAX_EPISODES = 10000
saver_path_base = 'reinforce_model/'
saver_path = None#'model/450'


class lstm_policy_network:
  def __init__(self, scope, trainable):
    # Assume default first input '0'
    #self.start_input = tf.zeros([1,1], tf.int32)
    self.start_input = 0
    self.scope = scope
    self.trainable = trainable
    # 0:<s>, 1-17:operand_idx, 18-29:unary_idx, 30-35:binary_idx
    with tf.variable_scope(self.scope):
      self.dsl_embeddings = tf.get_variable("dsl_embeddings", [sum(num_actions) + 1, embedding_size], trainable=self.trainable)
      self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
      self.Ws = []
      self.Bs = []
      for idx in range(MAX_LENGTH):
        block_idx = bisect.bisect_left([1,3,4], idx % 5)
        self.Ws.append(tf.get_variable("W_lstm_"+str(idx), [n_hidden_units, num_actions[block_idx]],
                   initializer=tf.random_normal_initializer(stddev=0.1), trainable=self.trainable))
        self.Bs.append(tf.get_variable("b_lstm_"+str(idx), [num_actions[block_idx]],
                   initializer=tf.constant_initializer(0), trainable=self.trainable))
  def get_parameters(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

  def _run_rnn(self, emb, init_state):
    return tf.nn.dynamic_rnn(self.lstm_cell, emb, initial_state=init_state, time_major=False, scope=self.scope)
  def init_state(self, batch_size):
    return self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
  
  def prob(self, target):
    '''output prob for a give target sequence
    Args:
      target: [batch_size, slen]
      target_gather: [batch_size, slen, 2], 2 is [batch_idx,target_elem]
    '''
    outprob = 1 # tf.fill(tf.shape(target[:,0:1]), 1.)
    input = tf.concat([tf.fill([1,1], self.start_input), target[:,:-1]], 1) # [1, slen]
    #input = tf.expand_dims(input, 0) # [1, slen, input_dim], when batchSize == 1
    emb = tf.nn.embedding_lookup(self.dsl_embeddings, input) # [1, slen, embedding_size]
    outputs, final_state = self._run_rnn(emb, self.init_state(batch_size))
    for idx in range(MAX_LENGTH):
      output_logtis = tf.matmul(outputs[:,idx], self.Ws[idx]) + self.Bs[idx] # [batch_size, num_action]
      outprobs = tf.nn.softmax(output_logtis)
      outprob *= outprobs[0,target[0, idx]]
    return outprob

  def multiple_step(self, input):
    input = tf.expand_dims(input, 0) # [1, slen, input_dim], assume batchSize == 1
    emb = tf.squeeze(tf.nn.embedding_lookup(self.dsl_embeddings, input), 2) # [1, slen, embedding_size]
    outputs, final_state = self._run_rnn(emb, self.init_state(1))
    for idx in range(MAX_LENGTH):
      #output_logit = output_logit.assign(tf.zeros([1 ,sum(num_actions) + 1]))
      #output_logit = tf.Variable(tf.zeros([1 ,sum(num_actions) + 1]), name="output_logit")
      block_idx = bisect.bisect_left([1,3,4], idx % 5)
      final_output = tf.matmul(outputs[:,idx], self.Ws[idx]) + self.Bs[idx]
      input_start = 1 + sum(num_actions[:block_idx])
      input_end = input_start + num_actions[block_idx]
      output_new = tf.concat([tf.constant(-np.inf, shape=[1,input_start]), final_output, tf.constant(-np.inf, shape=[1, sum(num_actions) + 1 - input_end])], 1)
      if(idx == 0):
        output_logits = output_new
      else:
        output_logits = tf.concat([output_logits, output_new], 0)
      #print("output_logits:", output_logits)
    print(output_logits)
    return output_logits

  def sample_multiple(self, input):
    '''
    output a sequence sampled from policy
    Return 
    '''
    #input = tf.constant(self.start_input, shape=[1,1], dtype=tf.int32)
    last_state = self.init_state(1)
    for idx in range(MAX_LENGTH):
      emb = tf.nn.embedding_lookup(self.dsl_embeddings, input) # [1, slen, embedding_size]
      outputs, last_state = self._run_rnn(emb, last_state)
      final_output = tf.matmul(outputs[:,0], self.Ws[idx]) + self.Bs[idx]
      predicted_action = tf.multinomial(final_output, 1)[:,0]
      input = tf.expand_dims(predicted_action, 0)
      if idx == 0:
        predicted_actions = predicted_action
      else:
        predicted_actions = tf.concat([predicted_actions, predicted_action],0)
    return predicted_actions
'''
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
    outputs, final_state = self._run_rnn(emb, self.init_state())
    for idx in range(MAX_LENGTH):
      #output_logit = output_logit.assign(tf.zeros([1 ,sum(num_actions) + 1]))
      #output_logit = tf.Variable(tf.zeros([1 ,sum(num_actions) + 1]), name="output_logit")
      block_idx = bisect.bisect_left([1,3,4], idx % 5)
      final_output = tf.matmul(outputs[:,idx], self.Ws[block_idx]) + self.Bs[block_idx]
      input_start = 1 + sum(num_actions[:block_idx])
      input_end = input_start + num_actions[block_idx]
      output_new = tf.concat([tf.constant(-np.inf, shape=[1,input_start]), final_output, tf.constant(-np.inf, shape=[1, sum(num_actions) + 1 - input_end])], 1)
      if(idx == 0):
        output_logits = output_new
      else:
        output_logits = tf.concat([output_logits, output_new], 0)
      #print("output_logits:", output_logits)
    print(output_logits)
    return output_logits

  def init_state(self):
    return self.lstm_cell.zero_state(1, dtype=tf.float32)

  # output a sequence sampled from policy
  def sample_multiple(self, input):
    #input = tf.expand_dims(input, 0) # [1, slen, input_dim], assume batchSize == 1
    input_print = tf.Print(input, [input], 'input shape')
    last_state = self.init_state()
    for idx in range(MAX_LENGTH):
      emb = tf.nn.embedding_lookup(self.dsl_embeddings, input_print) # [1, slen, embedding_size]
      outputs, last_state = self._run_rnn(emb, last_state)
      block_idx = bisect.bisect_left([1,3,4], idx % 5)
      final_output = tf.matmul(outputs[:,0], self.Ws[block_idx]) + self.Bs[block_idx]
      predicted_action = tf.multinomial(final_output, 1)[:,0]
      input_print = tf.expand_dims(predicted_action, 0)
      if idx == 0:
        predicted_actions = tf.Print(predicted_action, [predicted_action], 'predicted_action:')
      else:
        predicted_actions = tf.concat([predicted_actions, predicted_action],0)
    return predicted_actions
'''
pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       lstm_policy_network('policy',True),
                                       num_actions,
                                       n_hidden_units,
                                       saver_path=saver_path,
                                       summary_writer=writer)

'''
'''
def checkCode(code):
  for t in range(len(code)):
    block_idx = bisect.bisect_left([1,3,4], t % 5)
    assert code[t] <= num_actions[block_idx]

def dummyReward(code):
  return abs((code[0] - code[1]-(code[2] - code[3])) - code[4])

if __name__ == '__main__':
  best_code = []
  best_reward = -100
  episode_history = deque(maxlen=100)
  for i_episode in range(MAX_EPISODES):
    log_buffer = ''
    log_buffer += "Episode {}\n".format(i_episode)
    #print("Episode {}".format(i_episode))
    # initialize
    total_rewards = 0
    last_action = 0
    last_hidden = None
    code = []
    input = 0
    #pdb.set_trace()
    sequence = pg_reinforce.sampleSequence(np.array([[input]]))
    code = sequence
    log_buffer += str(code)+'\n'
    #print(code)
    reward = -(verifier.main(None, code))
    #reward = dummyReward(code)
    checkCode(code)
    for t in range(len(sequence)):
      block_idx = bisect.bisect_left([1,3,4], t % 5)
      action_index = sequence[t] + 1 + sum(num_actions[:block_idx])
      if t == MAX_LENGTH - 1:
        log_buffer += "reward:{}".format(reward)
        #print("reward:", reward)
        # if nan loss
        #if reward == -100:
        #  reward = -5
        # if loss is around same with initial loss
        #elif reward < -4.5:
        #  reward = -4.5
        pg_reinforce.storeRollout(input, last_action, reward)
      else:
        pg_reinforce.storeRollout(input, action_index, 0)
      input = action_index
      # if nan loss
      if reward == -100:
        reward = -5
      # if loss is around same with initial loss
      elif reward < -4.5:
        reward = -4.5

    '''
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
    '''
    pg_reinforce.updateModel()
    episode_history.append(reward)
    mean_rewards = np.mean(episode_history)
    if reward > best_reward:
          best_code = code
          best_reward = reward
    if i_episode % 100 == 0:
      pg_reinforce.save_model(saver_path_base+str(i_episode))
    if i_episode % 100 == 0:
      print(log_buffer)
      if i_episode % 10 == 0:
        print("Best code {} / rewards {}".format(', '.join([str(c) for c in best_code]), best_reward))
      print("code {}".format(', '.join([str(c) for c in code])))
      print("Reward for this episode: {}".format(reward))
      print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
  