import random
import numpy as np
import tensorflow as tf
import bisect

class PolicyGradientREINFORCE(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     num_actions,
                     lstm_hidden=150,
                     init_exp=0.5,         # initial exploration prob
                     final_exp=0.0,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_writer=None,
                     saver_path=None,
                     summary_every=100):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer

    # model components
    self.policy_network = policy_network

    # training parameters
    self.lstm_hidden     = lstm_hidden
    self.num_actions     = num_actions
    self.discount_factor = discount_factor
    self.max_gradient    = max_gradient
    self.reg_param       = reg_param

    # exploration parameters
    self.exploration  = init_exp
    self.init_exp     = init_exp
    self.final_exp    = final_exp
    self.anneal_steps = anneal_steps

    # counters
    self.train_iteration = 0

    # rollout buffer
    self.input_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []

    # record reward history for normalization
    self.all_rewards = []
    self.max_reward_length = 50#1000000

    self.saver = tf.train.Saver()
    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))
    if saver_path is not None:
      print("load from", saver_path)
      self.saver.restore(self.session, saver_path)

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

  def save_model(self, save_path):
    save_path = self.saver.save(self.session, save_path)
    print("Save to path: ", save_path)


  def resetModel(self):
    self.cleanUp()
    self.train_iteration = 0
    self.exploration     = self.init_exp
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

  def create_variables(self):

    with tf.name_scope("model_inputs"):
      # raw state representation
      self.input = tf.placeholder(tf.int32, (None, 1), name="input")
      self.inputs = tf.placeholder(tf.int32, (None, 1), name="inputs")
      self.idx = tf.placeholder(tf.int32, 1, name="idx")
      self.block_idx = tf.placeholder(tf.int32, 1, name="block_idx")
      #self.hidden = tf.placeholder(tf.float32, (None, self.lstm_hidden), name='hidden')
      #self.idx = 0
      self.hidden = self.policy_network.init_state()
    # rollout action based on current policy
    with tf.name_scope("predict_actions"):
      # initialize policy network
      with tf.variable_scope("policy_network"):
        self.policy_outputs = self.policy_network.step(self.input, self.idx, self.block_idx, self.hidden)


      # predict actions from policy network
      self.action_scores = tf.identity(self.policy_outputs[0], name="action_scores")
      #self.action_scores = tf.Print(self.action_scores, [self.action_scores, self.idx], "action_scores: ", summarize=30)
      self.hidden = self.policy_outputs[1]
      # Note 1: tf.multinomial is not good enough to use yet
      # so we don't use self.predicted_actions for now
      self.predicted_actions = tf.multinomial(self.action_scores, 1)

    # regularization loss
    policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

    # compute loss and gradients
    with tf.name_scope("compute_pg_gradients"):
      # gradients for selecting action from policy network
      self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
      self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

      with tf.variable_scope("policy_network", reuse=True):
        self.logprobs = self.policy_network.multiple_step(self.inputs)
        #self.logprobs = tf.Print(self.logprobs, [self.logprobs], "logprobs:", summarize=200)

      # compute policy loss and regularization loss
      self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs, labels=self.taken_actions)
      self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
      self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
      self.loss               = self.pg_loss + self.reg_param * self.reg_loss

      # compute gradients
      self.gradients = self.optimizer.compute_gradients(self.loss)
      
      # compute policy gradients
      for i, (grad, var) in enumerate(self.gradients):
        if grad is not None:
          self.gradients[i] = (grad * self.discounted_rewards, var)

      for grad, var in self.gradients:
        tf.summary.histogram(var.name, var)
        if grad is not None:
          tf.summary.histogram(var.name + '/gradients', grad)

      # emit summaries
      tf.summary.scalar("policy_loss", self.pg_loss)
      tf.summary.scalar("reg_loss", self.reg_loss)
      tf.summary.scalar("total_loss", self.loss)

    # training update
    with tf.name_scope("train_policy_network"):
      # apply gradients to update policy network
      self.train_op = self.optimizer.apply_gradients(self.gradients)

    self.summarize = tf.summary.merge_all()
    self.no_op = tf.no_op()

  def randomSampleAction(self, idx):
    block_range = [16,11,5]
    block_idx = bisect.bisect_left([1,3,4], idx % 5)
    return random.randint(0, block_range[block_idx])



  def sampleAction(self, input, idx):
    # TODO: use this code piece when tf.multinomial gets better
    # sample action from current policy
    # actions = self.session.run(self.predicted_actions, {self.input: input})[0]
    # return actions[0]

    # temporary workaround
    def softmax(y):
      """ simple helper function here that takes unnormalized logprobs """
      maxy = np.amax(y)
      e = np.exp(y - maxy)
      return e / np.sum(e)
    # epsilon-greedy exploration strategy
    if False and random.random() < self.exploration:
      return random.randint(0, self.num_actions[idx]-1)
    else:
      block_idx = bisect.bisect_left([1,3,4], idx % 5)
      action_scores = self.session.run(self.action_scores, {self.input: input, self.idx:[idx], self.block_idx: [block_idx]})[0]
      action_probs  = softmax(action_scores) - 1e-5
      action = np.argmax(np.random.multinomial(1, action_probs))
      return action

  def updateModel(self):

    N = len(self.reward_buffer)
    r = 0 # use discounted reward to approximate Q value

    # compute discounted future rewards
    discounted_rewards = np.zeros(1)
    assert sum(self.reward_buffer[:-1]) == 0
    discounted_rewards[:] = self.reward_buffer[-1]
    '''
    for t in reversed(range(N)):
      # future discounted reward from now on
      r = self.reward_buffer[t] + self.discount_factor * r
      discounted_rewards[t] = r
    '''
    # reduce gradient variance by normalization
    #self.all_rewards += discounted_rewards.tolist()
    self.all_rewards += [self.reward_buffer[-1]]
    self.all_rewards = self.all_rewards[-self.max_reward_length:]
    #print self.all_rewards
    discounted_rewards -= np.mean(self.all_rewards)
    #discounted_rewards /= (np.std(self.all_rewards) + 1)

    # whether to calculate summaries
    calculate_summaries = self.summary_writer is not None and self.train_iteration % self.summary_every == 0

    # update policy network with the rollout in batches
    # prepare inputs
    inputs  = np.expand_dims(np.array(self.input_buffer), axis=1)
    #print("inputs: ", inputs)
    actions = np.array(self.action_buffer)
    #print("actions: ", actions)
    rewards = discounted_rewards

    # evaluate gradients
    grad_evals = [grad for grad, var in self.gradients]

    # perform one update of training
    _, summary_str = self.session.run([
      self.train_op,
      self.summarize if calculate_summaries else self.no_op
    ], {
      self.inputs:             inputs,
      self.taken_actions:      actions,
      self.discounted_rewards: rewards,
    })

    # emit summaries
    if calculate_summaries:
      self.summary_writer.add_summary(summary_str, self.train_iteration)

    self.annealExploration()
    self.train_iteration += 1

    # clean up
    self.cleanUp()

  def annealExploration(self, stategy='linear'):
    ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
    self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

  def storeRollout(self, input, action, reward):
    self.action_buffer.append(action)
    self.reward_buffer.append(reward)
    self.input_buffer.append(input)

  def cleanUp(self):
    self.input_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []