"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import bisect
from collections import deque
import cifar10_train as verifier
import os
import random
import math
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EP_LEN = 1
EP_MAX = 30000
A_LR = 0.0001
A_DIM = 5
A_UPDATE_STEPS = 10
MAX_LENGTH = 5
num_actions = [17,12,6]
#OLD_UPDATE_STEPS = 5
METHOD = dict(name='clip', epsilon=0.2)
SAVE_PATH='Saved_PPO/cifar10_REINFORCE_'
LOAD_PATH=''#Saved_PPO/Dummy_step_200_avgR_7.6'
print('EP_LEN',EP_LEN,', EP_MAX',EP_MAX, 'A_UPDATE_STEPS', A_UPDATE_STEPS)
'''
import tensorflow as tf
class Object(object):
	pass

self = Object()
'''

class PPO(object):

	def __init__(self):
		self.sess = tf.Session()
		# critic
		with tf.variable_scope('critic'):
			#self.moving_average = -4.0
			self.tf_r = tf.placeholder(tf.float32, [None, 1], 'reward')
			self.tf_r_1 = tf.placeholder(tf.float32, [], 'single_reward')
			ema = tf.train.ExponentialMovingAverage(decay=0.9, zero_debias=True)
			self.update_critic = ema.apply([self.tf_r_1])
			self.moving_average = ema.average(self.tf_r_1)
			#self.moving_average = tf.Print(self.moving_average, [self.moving_average], 'Moving Average: ')
			self.advantage = self.tf_r - self.moving_average
		# actor
		new_policy = lstm_policy_network('new_policy', True)
		#old_policy = lstm_policy_network('old_policy', False)
		
		#self.input_codes = tf.placeholder(tf.int32, [None, None], 'input_codes')
		self.input_codes = tf.placeholder_with_default(tf.constant([[1,2,3,4,5]]), shape=[None, None])
		self.use_input_codes = tf.placeholder_with_default(tf.constant(0), shape=None)
		self.batch_size = tf.placeholder_with_default(tf.constant(1), shape=[])
		sample_input = tf.cond(tf.equal(self.use_input_codes, 1), lambda:self.input_codes, lambda:tf.expand_dims(tf.constant([1,2,3,4,5]), 0))
		self.sample_opt_op = new_policy.sample_sequence(sample_input, self.batch_size)
		#self.old_sample_op = old_policy.sample_sequence(self.input_codes, self.batch_size)
		self.target_gather = tf.placeholder(tf.int32, [None,A_DIM,2])
		self.tfa = tf.placeholder(tf.int32, [None, A_DIM], 'action')
		self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

		with tf.variable_scope('loss'):
			with tf.variable_scope('surrogate'):
				ratio = new_policy.prob(self.tfa, self.target_gather, self.batch_size, self.input_codes) #/ old_policy.prob(self.tfa, self.target_gather, self.batch_size, self.input_codes)
				#ratio = tf.Print(ratio,[ratio],'ratio: ',summarize=100)
				surr = ratio * self.tfadv
			if METHOD['name'] == 'clip':
				#self.aloss = -tf.reduce_mean(tf.minimum(
				#	surr,
				#	tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))
				self.aloss = -tf.reduce_mean(surr)
		new_policy_params  = new_policy.get_parameters()
		#old_policy_params = old_policy.get_parameters()
		#self.update_old_policy_op = [oldp.assign(p) for p, oldp in zip(new_policy_params, old_policy_params)]

		with tf.variable_scope('atrain'):
			self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss, var_list=new_policy_params)
		#print(new_policy_params)
		#print(tf.trainable_variables())
		tf.summary.FileWriter("log/", self.sess.graph)

		self.saver = tf.train.Saver()
		if LOAD_PATH != '':
			self.load(LOAD_PATH)
		else:
			self.sess.run(tf.global_variables_initializer())

	def update(self, a, r, step, input_codes):
		batch_size = len(a)
		a_1 = np.repeat(np.array([i for i in range(batch_size)])[:,None], A_DIM, axis=1)
		a_gather_ = np.concatenate((np.expand_dims(a_1, axis=2), np.expand_dims(a, axis=2)), axis=2)

		for idx in range(MAX_LENGTH):
			block_idx = bisect.bisect_left([1,3,4], idx % 5)
		  	a[:,idx] += sum(num_actions[:block_idx])
		# r: float, a: [batch_size, slen]
		#if step % OLD_UPDATE_STEPS == 0:
		#self.sess.run(self.update_old_policy_op)
		adv = self.sess.run(self.advantage, {self.tf_r: r})
		#print("advantage: ", adv)
		# adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

		# update actor
		if METHOD['name'] == 'clip':
			 # clipping method, find this is better (OpenAI's paper)
			 [self.sess.run(self.atrain_op, {self.tfa: a, self.tfadv: adv, self.batch_size: batch_size, self.target_gather: a_gather_, self.input_codes:input_codes, self.use_input_codes:1}) for _ in range(A_UPDATE_STEPS)]

		# update critic
		for r_ in r:
			self.sess.run(self.update_critic, {self.tf_r_1: r_[0]})
		#self.moving_average = self.moving_average * 0.99 + r * 0.01
	def save(self, step, avg_reward):
		saver_path = SAVE_PATH + 'step_' + str(step) + '_avgR_' + '{:.2}'.format(avg_reward)
		print("saving to", saver_path)
		self.saver.save(self.sess, saver_path)

	def load(self, saver_path):
		print("loading from ", saver_path)
		self.saver.restore(self.sess, saver_path)
		

	def _build_anet(self, name, trainable):
		with tf.variable_scope(name):
			l1 = tf.layers.dense(100, tf.nn.relu, trainable=trainable)
			mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
			sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
			norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
		return norm_dist, params

	def choose_sequence(self, input_codes):
		return self.sess.run(self.sample_opt_op)#, {self.input_codes:input_codes, self.batch_size:len(input_codes)})


'''
from my_PPO import lstm_policy_network_test
import numpy as np
testc = lstm_policy_network_test()
slen = 5
batch_size = 2
target_ = np.array([[1,2,3,4,5], [2,3,4,1,5]])
target_gather_1 = np.repeat(np.array([i for i in range(batch_size)])[:,None], slen, axis=1)
target_gather_ = np.concatenate((np.expand_dims(target_gather_1, axis=2), np.expand_dims(target_, axis=2)), axis=2)
testc.prob_test(target_, target_gather_) 
testc.sample_seq_test()
testc.policy.get_parameters()

'''
class lstm_policy_network_test:
		def __init__(self):
			self.policy = lstm_policy_network('test_policy', False)
			self.target = tf.placeholder(tf.int32, [None,5])
			self.target_gather = tf.placeholder(tf.int32, [None,5,2])
			self.batch_size = tf.placeholder(tf.int32, [])
			self.prob_op = self.policy.prob(self.target, self.target_gather, self.batch_size)
			self.sample_seq_op = self.policy.sample_sequence()
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
			print("variables initialized")
		def prob_test(self, target, target_gather):
			return self.sess.run(self.prob_op, {self.target:target, self.target_gather:target_gather, self.batch_size:len(target)})
		def sample_seq_test(self):
			return self.sess.run(self.sample_seq_op)

class lstm_policy_network:
	def __init__(self, scope, trainable):
		n_hidden_units = 150
		embedding_size = 100
		MAX_LENGTH    = 5
		self.epsilon = 1e-8
		# Assume default first input '0'
		#self.start_input = tf.zeros([1,1], tf.int32)
		self.start_input = sum(num_actions)
		self.scope = scope
		self.trainable = trainable
		# 0:<s>, 1-17:operand_idx, 18-29:unary_idx, 30-35:binary_idx
		with tf.variable_scope(self.scope):
			self.encoder = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
			self.encoder_embedding = tf.get_variable("encoder_embeddings", [6 + 1, embedding_size], trainable=self.trainable)
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
	
	def encode(self, input_codes, batch_size):
		# input_codes [batch_size, slen]
		emb = tf.nn.embedding_lookup(self.encoder_embedding, input_codes)
		encoder_outputs, encoder_states = self._run_rnn(self.encoder, emb, self.init_state(batch_size),'encoder')
		return encoder_outputs, encoder_states
	
	def get_parameters(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

	def _run_rnn(self, lstm_cell, emb, init_state, scope):
		with tf.variable_scope(self.scope):
			return tf.nn.dynamic_rnn(lstm_cell, emb, initial_state=init_state, time_major=False, scope=scope)

	def init_state(self, batch_size):
		return self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
	
	def prob(self, target, target_gather, batch_size, input_codes):
		'''output prob for a give target sequence
		Args:
			target: [batch_size, slen]
			target_gather: [batch_size, slen, 2], 2 is [batch_idx,target_elem]
		'''
		encoder_outputs, encoder_states = self.encode(input_codes, batch_size)
		#encoder_states = self.init_state(batch_size)
		outprob = 1 # tf.fill(tf.shape(target[:,0:1]), 1.)
		input = tf.concat([tf.fill(tf.shape(target[:,0:1]), self.start_input), target[:,:-1]], 1) # [1, slen]
		#input = tf.expand_dims(input, 0) # [1, slen, input_dim], when batchSize == 1
		emb = tf.nn.embedding_lookup(self.dsl_embeddings, input) # [1, slen, embedding_size]
		outputs, final_state = self._run_rnn(self.lstm_cell, emb, encoder_states,'decoder')
		for idx in range(MAX_LENGTH):
			output_logtis = tf.matmul(outputs[:,idx], self.Ws[idx]) + self.Bs[idx] # [batch_size, num_action]
			outprobs = tf.nn.softmax(output_logtis)
			#outprob *= outprobs[0,target[0, idx]]
			outprob += tf.log(tf.gather_nd(outprobs, target_gather[:,idx,:]) + self.epsilon)
			#outprob *= tf.gather_nd(outprobs, target_gather[:,idx,:])
		return outprob
	'''
	outprobs [batch_size, num_action]
	target [batch_size, slen]
	target[:,idx] [batch_size]
	target_gather [batch_size, slen, 2]
	target_gather[:,idx] [batch_size, 2]

	outprobs [2, 2]
	target [2, 2]
	*= [2, 1]
	'''
	def sample_sequence(self, input_codes, batch_size):
		'''
		output a sequence sampled from policy
		Return 
		'''
		encoder_outputs, encoder_states = self.encode(input_codes, batch_size)
		input = tf.fill([batch_size,1], self.start_input)
		#input = tf.constant(self.start_input, shape=[batch_size,1], dtype=tf.int32)
		last_state = encoder_states
		#last_state = self.init_state(1)
		for idx in range(MAX_LENGTH):
		  block_idx = bisect.bisect_left([1,3,4], idx % 5)
		  input = input + sum(num_actions[:block_idx]) # make input globally indexed
		  emb = tf.nn.embedding_lookup(self.dsl_embeddings, input) # [batch_size, 1, embedding_size]
		  outputs, last_state = self._run_rnn(self.lstm_cell, emb, last_state,'decoder')
		  final_output = tf.matmul(outputs[:,0], self.Ws[idx]) + self.Bs[idx]
		  predicted_action = tf.multinomial(final_output, 1)
		  #input = tf.expand_dims(predicted_action, 1)
		  if idx == 0:
			predicted_actions = predicted_action
		  else:
			predicted_actions = tf.concat([predicted_actions, predicted_action],1)
		return predicted_actions

def dummyReward(code):
  return abs((code[0] - code[1]-(code[2] - code[3])) - code[4])

def dummyRewardWInput(code, score):
	return -abs(abs((code[0] - code[1]-(code[2] - code[3])) - code[4]) - score)

def dummyInputScore(input_codes):
	# input_codes [78 number]
	list_num = [i for i in range(1,13)]
	structured_seq = []
	global_idx = 0
	for num in list_num:
		structured_seq += [input_codes[global_idx : global_idx+num]]
		global_idx += num
	score = 0
	for slice_idx, slice in enumerate(structured_seq):
		if slice_idx > 0:
			score += math.pow(-1, slice_idx) * sum(slice[1:]) % slice[0]
	return score

def sample_input_codes():
	rand1 = lambda: random.randint(2,5)
	rand2 = lambda: random.randint(0,1)
	seq = []
	for i in range(0,12):
		seq += [rand1()]
		for j in range(i):
			seq += [rand2()]
	return seq

class train_optimizer_controller():
	def __init__(self):
		self.ppo = PPO()
		self.reset()
	def reset(self):
		self.best_code = []
		self.best_score = 0
		self.best_reward = -100
		self.episode_history = deque(maxlen=100)
		self.global_idx = 0

	def train(self, verifier):
		for ep in range(EP_MAX):
			buffer_sequences, buffer_reward = [], []
			input_codes = []
			for t in range(EP_LEN):
				input_codes += [sample_input_codes()]
			sampled_sequences = ppo.choose_sequence(input_codes)
			for t in range(EP_LEN):
				score = dummyInputScore(input_codes[t])
				sampled_sequence = sampled_sequences[t]
				reward = dummyRewardWInput(sampled_sequence, score)
				buffer_sequences.append(sampled_sequence)
				buffer_reward.append(reward)
				if t == EP_LEN - 1:
					#pdb.set_trace()
					batch_sequences, batch_rewards = np.vstack(buffer_sequences), np.vstack(buffer_reward)
					input_codes = np.array(input_codes)
					buffer_sequences, buffer_reward = [], []
					ppo.update(batch_sequences, batch_rewards, ep, input_codes)
				episode_history.append(reward)
				mean_rewards = np.mean(episode_history)
				if reward > best_reward:
					best_code = sampled_sequence
					best_score = score
					best_reward = reward


if __name__ == '__main__':
	best_code = []
	best_score = 0
	best_reward = -100
	ppo = PPO()
	episode_history = deque(maxlen=100)
	global_idx = 0
	for ep in range(EP_MAX):
		buffer_sequences, buffer_reward = [], []
		input_codes = []
		for t in range(EP_LEN):
			input_codes += [sample_input_codes()]
		sampled_sequences = ppo.choose_sequence(input_codes)
		for t in range(EP_LEN):
			score = dummyInputScore(input_codes[t])
			sampled_sequence = sampled_sequences[t]
			reward = dummyRewardWInput(sampled_sequence, score)
			buffer_sequences.append(sampled_sequence)
			buffer_reward.append(reward)
			if t == EP_LEN - 1:
				#pdb.set_trace()
				batch_sequences, batch_rewards = np.vstack(buffer_sequences), np.vstack(buffer_reward)
				input_codes = np.array(input_codes)
				buffer_sequences, buffer_reward = [], []
				ppo.update(batch_sequences, batch_rewards, ep, input_codes)
			episode_history.append(reward)
			mean_rewards = np.mean(episode_history)
			if reward > best_reward:
				best_code = sampled_sequence
				best_score = score
				best_reward = reward

		'''
		for t in range(EP_LEN):
			sampled_sequence = ppo.choose_sequence(input_codes)
			reward = dummyReward(sampled_sequence)
			' ''
			reward = -(verifier.main(None, sampled_sequence))
			# if nan loss
			if reward == -100:
				reward = -5
			# if loss is around same with initial loss
			elif reward < -4.5:
				reward = -4.5
			' ''
			buffer_sequences.append(sampled_sequence)
			buffer_reward.append(reward)
			if t == EP_LEN - 1:
				batch_sequences, batch_rewards = np.vstack(buffer_sequences), np.vstack(buffer_reward)
				buffer_sequences, buffer_reward = [], []
				ppo.update(batch_sequences, batch_rewards, ep, input_codes)
			# log keeper
			episode_history.append(reward)
			mean_rewards = np.mean(episode_history)
			if reward > best_reward:
				best_code = sampled_sequence
				best_reward = reward
		'''
		if global_idx != 0 and global_idx % 100 == 0: 
			print(
				'Ep: %i' % global_idx,
				"|reward: %i" % reward,
				"|sequence: %s" % str(sampled_sequence)
			)
			print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
			if global_idx % 10 == 0:
				print("Best code {} | {} / rewards {}".format(', '.join([str(c) for c in best_code]), best_score, best_reward))
			#if global_idx % 100 == 0:
			#	ppo.save(global_idx, mean_rewards)

		global_idx += 1
