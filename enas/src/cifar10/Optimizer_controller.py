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
from tensorflow.python.ops import state_ops
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import bisect
from collections import deque
import os
import random
import math
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EP_LEN = 3
EP_MAX = 1000
A_LR = 1e-2
A_UPDATE_STEPS = 1
A_UPDATE_STEPS_OLD = 1
MAX_LENGTH = 20 + 1
A_DIM = MAX_LENGTH
#num_actions = [17,17,12,12,6,7]
num_actions = [[6+i, 6, 6] for i in range(4)] # 4 group of actions
num_actions[-1].append(7) # append lr choice to last group
#num_actions = [num for sublist in num_actions for num in sublist]
n_hidden_units = 150
embedding_size = 100
#OLD_UPDATE_STEPS = 5
METHOD = dict(name='clip', epsilon=0.2)
SAVE_PATH='Saved_PPO/cifar10_REINFORCE_'
LOAD_PATH=''#Saved_PPO/Dummy_step_200_avgR_7.6'
'''
import tensorflow as tf
class Object(object):
	pass

self = Object()
'''

class PPO(object):

	def __init__(self, sample_arc_op):
		self.entropy_weight = 0.0015
		self.arc_var_weight = 0.0015
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
		old_policy = lstm_policy_network('old_policy', False)
		self.input_codes = tf.placeholder_with_default(tf.constant([[1,2,3,4,5]]), shape=[None, None])
		self.use_input_codes = tf.placeholder_with_default(tf.constant(0), shape=None)
		self.batch_size = tf.placeholder_with_default(tf.constant(1), shape=[])
		sample_input = tf.cond(tf.equal(self.use_input_codes, 1), lambda:self.input_codes, lambda:tf.expand_dims(sample_arc_op, 0))
		self.sample_opt_op = new_policy.sample_sequence(sample_input, self.batch_size)
		self.target_gather = tf.placeholder(tf.int32, [None,A_DIM,2])
		self.tfa = tf.placeholder(tf.int32, [None, A_DIM], 'action')
		self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
		print('EP_LEN:',EP_LEN,', EP_MAX:',EP_MAX, 'A_UPDATE_STEPS:', A_UPDATE_STEPS, 'A_LR:', A_LR)
		print('opt controller optimizer: PPO, entropy_weight {}, self.arc_var_weight: {}'.format(self.entropy_weight, self.arc_var_weight))
		#with tf.variable_scope('loss'):
			#with tf.variable_scope('surrogate'):
		prob_new, entropy, var = new_policy.prob(self.tfa, self.target_gather, self.batch_size, self.input_codes) 
		prob_old, _, _ = old_policy.prob(self.tfa, self.target_gather, self.batch_size, self.input_codes) 
		ratio = tf.exp(prob_new - prob_old)
		pg_loss1 = -self.tfadv*ratio
		pg_loss2 = -tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv
		pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
		#pg_loss = -self.tfadv*prob_new
		self.aloss = pg_loss - entropy * self.entropy_weight - var * self.arc_var_weight
		'''
				ratio, entropy = new_policy.prob(self.tfa, self.target_gather, self.batch_size, self.input_codes) 
				surr = ratio * self.tfadv + entropy * self.entropy_weight# + var * self.arc_var_weight
			if METHOD['name'] == 'clip':
				self.aloss = -tf.reduce_mean(surr)
		'''
		new_policy_params  = new_policy.get_parameters()
		old_policy_params = old_policy.get_parameters()
		self.update_old_policy_op = [oldp.assign(p) for p, oldp in zip(new_policy_params, old_policy_params)]
		
		with tf.variable_scope('atrain'):
			self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss, var_list=new_policy_params)

	def update(self, a, r, step, input_codes, sess):
		batch_size = len(a)
		a_1 = np.repeat(np.array([i for i in range(batch_size)])[:,None], A_DIM, axis=1)
		a_gather_ = np.concatenate((np.expand_dims(a_1, axis=2), np.expand_dims(a, axis=2)), axis=2)

		for idx in range(MAX_LENGTH):
			block_idx = bisect.bisect_left([1,3,4], idx % 5)
			a[:,idx] += sum(num_actions[-1][:block_idx])
		# r: float, a: [batch_size, slen]
		#sess.run(self.update_old_policy_op)
		if step % A_UPDATE_STEPS_OLD == 0:
			sess.run(self.update_old_policy_op)
		adv = sess.run(self.advantage, {self.tf_r: r})
		#print("advantage: ", adv)
		# adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

		# update actor
		if METHOD['name'] == 'clip':
			 # clipping method, find this is better (OpenAI's paper)
			 [sess.run(self.atrain_op, {self.tfa: a, self.tfadv: adv, self.batch_size: batch_size, self.target_gather: a_gather_, self.input_codes:input_codes, self.use_input_codes:1}) for _ in range(A_UPDATE_STEPS)]

		# update critic
		for r_ in r:
			sess.run(self.update_critic, {self.tf_r_1: r_[0]})
		#self.moving_average = self.moving_average * 0.99 + r * 0.01

	def choose_sequence(self, input_codes, sess):
		return sess.run(self.sample_opt_op, {self.input_codes:input_codes, self.batch_size:len(input_codes), self.use_input_codes:1})


class lstm_policy_network:
	def __init__(self, scope, trainable):
		self.epsilon = 1e-8
		# Assume default first input '0'
		#self.start_input = tf.zeros([1,1], tf.int32)
		self.start_input = sum(num_actions[-1])
		self.scope = scope
		self.trainable = trainable
		# 0:<s>, 1-17:operand_idx, 18-29:unary_idx, 30-35:binary_idx
		#self.mean_probs = [num_actions[i][j] for i ]
		with tf.variable_scope(self.scope):
			self.encoder = tf.contrib.rnn.LSTMCell(n_hidden_units, initializer=tf.random_uniform_initializer(-0.08, 0.08), forget_bias=1.0, state_is_tuple=True)
			# encoding for arc, 6 nodes in total
			self.encoder_embedding = tf.get_variable("encoder_embeddings", [6 + 1, embedding_size], trainable=self.trainable)
			self.dsl_embeddings = tf.get_variable("dsl_embeddings", [sum(num_actions[-1]) + 1, embedding_size], trainable=self.trainable, initializer=tf.random_uniform_initializer(-0.08, 0.08))
			self.lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden_units, initializer=tf.random_uniform_initializer(-0.08, 0.08), forget_bias=1.0, state_is_tuple=True)
			self.Ws = []
			self.Bs = []
			self.AvgProbs = []
			for idx in range(MAX_LENGTH):
				# if idx is last idx -> predicting learning rate
				if idx == MAX_LENGTH - 1:
					dim = num_actions[-1][-1]
				else:
					block_idx = bisect.bisect_left([1,3,4], idx % 5)
					dim = num_actions[idx//5][block_idx]
				self.Ws.append(tf.get_variable("W_lstm_"+str(idx), [n_hidden_units, dim], dtype=tf.float32,
								   initializer=tf.random_uniform_initializer(-0.08, 0.08), trainable=self.trainable))
				self.Bs.append(tf.get_variable("b_lstm_"+str(idx), [dim], dtype=tf.float32,
								   initializer=tf.constant_initializer(0), trainable=self.trainable))
				self.AvgProbs.append(tf.get_variable("AvgProbs_"+str(idx), [EP_LEN, dim], dtype=tf.float32,
								   initializer=tf.random_uniform_initializer(-0.08, 0.08), trainable=False))
			#print("self.Bs: ", self.Bs)
	
	def encode(self, input_codes, batch_size):
		# input_codes [batch_size, slen]
		emb = tf.nn.embedding_lookup(self.encoder_embedding, input_codes)
		encoder_outputs, encoder_states = self._run_rnn(self.encoder, emb, self.init_state(batch_size),'encoder')
		return encoder_outputs, encoder_states
	
	def get_parameters(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

	def _run_rnn(self, lstm_cell, emb, init_state, scope):
		with tf.variable_scope(self.scope, initializer=tf.random_uniform_initializer(-0.08, 0.08)):
			return tf.nn.dynamic_rnn(lstm_cell, emb, initial_state=init_state, time_major=False, scope=scope)

	def init_state(self, batch_size):
		return self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
	
	def prob(self, target, target_gather, batch_size, input_codes):
		'''output prob for a give target sequence
		Args:
			target: [batch_size, slen]
			target_gather: [batch_size, slen, 2], 2 is [batch_idx,target_elem]
		'''
		#single_batch_assert = tf.Assert(tf.equal(batch_size, 1), [batch_size])
		#with tf.control_dependencies([single_batch_assert]):
		encoder_outputs, encoder_states = self.encode(input_codes, batch_size)
		#encoder_states = self.init_state(batch_size)
		outprob = 0 # tf.fill(tf.shape(target[:,0:1]), 1.)
		outentropy = 0
		diff_arc_var = 0
		input = tf.concat([tf.fill(tf.shape(target[:,0:1]), self.start_input), target[:,:-1]], 1) # [1, slen]
		#input = tf.expand_dims(input, 0) # [1, slen, input_dim], when batchSize == 1
		emb = tf.nn.embedding_lookup(self.dsl_embeddings, input) # [1, slen, embedding_size]
		outputs, final_state = self._run_rnn(self.lstm_cell, emb, encoder_states,'decoder')
		for idx in range(MAX_LENGTH):
			output_logtis = tf.matmul(outputs[:,idx], self.Ws[idx]) + self.Bs[idx] # [batch_size, num_action]
			outprobs = tf.nn.softmax(output_logtis)
			#mean_probs = self.AvgProbs[idx]
			#mean_probs = tf.fill(tf.shape(outprobs),tf.reduce_mean(outprobs))
			diff_arc_var += tf.reduce_mean(tf.square(outprobs - self.AvgProbs[idx]))
			state_ops.assign(self.AvgProbs[idx], 0.9 * self.AvgProbs[idx] + 0.1 * tf.reduce_mean(outprobs))
			#outprob *= outprobs[0,target[0, idx]]
			#outentropy += tf.nn.softmax_cross_entropy_with_logits(labels=outprobs, logits=output_logtis)
			outentropy += -tf.reduce_sum(outprobs * tf.log(outprobs), 1)
			outprob += tf.log(tf.gather_nd(outprobs, target_gather[:,idx,:]) + self.epsilon)
			#outprob *= tf.gather_nd(outprobs, target_gather[:,idx,:])
		return outprob, outentropy, diff_arc_var
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
		single_batch_assert = tf.Assert(tf.equal(batch_size, 1), [batch_size])
		encoder_outputs, encoder_states = self.encode(input_codes, batch_size)
		with tf.control_dependencies([single_batch_assert]):
			input = tf.fill([batch_size,1], self.start_input)
		#input = tf.constant(self.start_input, shape=[batch_size,1], dtype=tf.int32)
		last_state = encoder_states
		#last_state = self.init_state(1)
		for idx in range(MAX_LENGTH):
		  if(idx > 0):
		  	block_idx = bisect.bisect_left([1,3,4], (idx - 1) % 5)
		  	input = input + sum(num_actions[-1][:block_idx]) # make input globally indexed
		  emb = tf.nn.embedding_lookup(self.dsl_embeddings, input) # [batch_size, 1, embedding_size]
		  outputs, last_state = self._run_rnn(self.lstm_cell, emb, last_state,'decoder')
		  final_output = tf.matmul(outputs[:,0], self.Ws[idx]) + self.Bs[idx]
		  # if second operand
		  if(idx % 5 == 1):
		  	# force the left and right operands to be different at each iteration
		  	#final_output[:,predicted_action] = tf.fill(tf.shape(final_output[:,0:1]),-np.inf)
		  	print(final_output[:,:predicted_action[0][0]])#, final_output[:,predicted_action+1:])
		  	final_output = tf.concat((final_output[:,:predicted_action[0][0]], tf.fill(tf.shape(final_output[:,0:1]),-np.inf), final_output[:,predicted_action[0][0]+1:]), axis=1)
		  	# force to reuse one of the previously computed operand
		  	if idx // 5 > 0:
		  		final_output = tf.cond(tf.less(predicted_action[0][0], num_actions[0][0]),
												lambda: tf.concat((tf.fill(tf.shape(final_output[:,:num_actions[0][0]]), -np.inf), final_output[:,num_actions[0][0]:]), axis=1),
												lambda: final_output
												)
		  predicted_action = tf.multinomial(final_output, 1)
		  input = predicted_action
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

def dummy_train_opt():
	def _dummy_verifier(input_codes, opt_sequence):
		'''
		Args
			input_codes: [78]
			opt_sequence: [5]
		Return
			ratio [0, 1)
		'''
		input_total = sum(input_codes)
		opt_total = sum(opt_sequence) * opt_sequence[0]
		epsilon = 1e-8
		return min(input_total, opt_total) / (max(input_total, opt_total) + epsilon)
	def _sample_arc():
		logits = tf.log([[1.,1.,1.,1.,1.]])
		lst = []
		for nlayer in range(1, 13):
			logits = tf.log([[1.] * nlayer])
			number = tf.multinomial(logits, 1)
			number = tf.to_int32(number)
			lst.append(number[0][0])
			for prec_layer in range(1, nlayer):
				logits = tf.log([[1., 1.]])
				number = tf.multinomial(logits, 1)
				number = tf.to_int32(number)
				lst.append(number[0][0])
		lst = tf.stack(lst, axis=0)
		return tf.reshape(lst, [-1])
	opt_controller = train_optimizer_controller(_sample_arc())
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		opt_controller.train(_dummy_verifier, sess)

class train_optimizer_controller():
	def __init__(self, sample_arc_op):
		self.ppo = PPO(sample_arc_op)
		self.sample_arc_op = sample_arc_op
		self.reset()
	def reset(self):
		self.best_code = []
		self.best_score = 0
		self.best_reward = -1e8
		self.episode_history = deque(maxlen=20)
		self.input_history, self.opt_history = deque(maxlen=20), deque(maxlen=20)
		self.global_idx = 0

	def codes_distance(self, codes):
		codes_array = np.array(codes)
		mc = stats.mode(codes_array)[0]
		mcs = np.repeat(mc, len(codes), axis=0)
		return np.sum(np.not_equal(codes_array, mcs)) / ((codes_array.shape[0] - 1) * codes_array.shape[1] + 1e-8)

	def train(self, verifier, sess):
		for ep in range(EP_MAX):
			buffer_sequences, buffer_reward = [], []
			input_codes, sampled_sequences = [], []
			sample_arc_one = sess.run(self.sample_arc_op)
			sample_seq_one = self.ppo.choose_sequence([sample_arc_one], sess)[0]
			for t in range(EP_LEN):
				input_codes += [sample_arc_one]#[sess.run(self.sample_arc_op)]
				sampled_sequences += [sample_seq_one]#[self.ppo.choose_sequence([input_codes[-1]], sess)[0]]
			#sampled_sequences = self.ppo.choose_sequence(input_codes, sess)
			for t in range(EP_LEN):
				#score = dummyInputScore(input_codes[t])
				self.global_idx += 1
				sampled_sequence = sampled_sequences[t]
				reward = verifier(input_codes[t], sampled_sequence)
				#reward = dummyRewardWInput(sampled_sequence, score)
				buffer_sequences.append(sampled_sequence)
				buffer_reward.append(reward)
				if t == EP_LEN - 1:
					#pdb.set_trace()
					self.input_history.extend(input_codes)
					self.opt_history.extend(buffer_sequences)
					batch_sequences, batch_rewards = np.vstack(buffer_sequences), np.vstack(buffer_reward)
					input_codes = np.array(input_codes)
					buffer_sequences, buffer_reward = [], []
					self.ppo.update(batch_sequences, batch_rewards, ep, input_codes, sess)
				self.episode_history.append(reward)
				mean_rewards = np.mean(self.episode_history)
				if reward > self.best_reward:
					self.best_code = sampled_sequence
					self.best_reward = reward
			if self.global_idx % 10 == 0: 
				print(
					'Ep: %i' % self.global_idx,
					", Last 20 episodes| avg reward: %f" % mean_rewards,
					"|architecture distance: %0.2f" % self.codes_distance(self.input_history),
					"|opt codes distance: %0.2f" % self.codes_distance(self.opt_history)
				)
				#if self.global_idx % 10 == 0:
				#	print("Last 20 episodesAverage reward for last 20 episodes: {:.2f}".format(mean_rewards))
					#print("Best code {} | {} / rewards {}".format(', '.join([str(c) for c in self.best_code]), self.best_score, self.best_reward))

	def sample_opt_op(self):
		return self.ppo.sample_opt_op[0]

if __name__ == '__main__':
	dummy_train_opt()