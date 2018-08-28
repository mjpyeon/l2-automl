import math
import pdb
import os
import sys
import datetime
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from args import args
from tensorboardX import SummaryWriter
from datetime import datetime
import logger as L
from args import args

class Saver:
	'''
	class for saving and loading optimizee and saving tensorboard information
	'''
	def __init__(self, save_dir):
		self.logger = L.get_logger(args.logdir)
		current_time = datetime.now().strftime('%b%d_%H-%M-%S')
		self.tensorboard_writer = SummaryWriter(log_dir=save_dir+'/tensorboard_'+current_time)
		self.logger = self.logger
		self.n_iter_scalar = 1
		self.n_iter_embedding = 1
		self.betas_embedding = {}
		self.beta_grads = {}
		self.save_dir = save_dir

	def save_checkpoint(self, optimizee, epoch, episode, filepath):
		"""
		Save checkpoint of optimizee.model and meta_optim states including beta (and alpha if using darts arch)
		"""
		self.logger.info("=> saving checkpoint '{}'".format(filepath))
		state = {'episode': episode, 'epoch':int(epoch) + 1, 'optimizee':optimizee.state_dict()}
		torch.save(state, filepath)

	def save_beta_states(self, beta_states, filename):
		if(not os.path.isdir(self.save_dir)):
			os.makedirs(self.save_dir)
		if(beta_states is not None):
			save_path = self.save_dir + '/beta_best_' + str(filename)
			self.logger.info('=> save beta to {}'.format(save_path))
			torch.save(beta_states, save_path)	

	def save_beta(self, meta_optim, n_iter):
		if(not os.path.isdir(self.save_dir)):
			os.makedirs(self.save_dir)
		save_path = self.save_dir + '/beta_' + str(n_iter)
		self.logger.info('=> save beta to {}'.format(save_path))
		torch.save(meta_optim.state_dict(NoStates=True), save_path)

	def load_beta(self, meta_optim, beta_states):
		"""
		Loads saved beta to given meta_optim
		"""
		meta_optim.load_state_dict(beta_states, NoStates=True)
		#self.logger.info("=> loaded beta/lr from '{}'".format(filepath))

	def load_alpha(self, optimizee, alpha):
		"""
		Loads saved alpha to given optimizee
		"""
		optimizee.model.set_arch_paramters(alpha)
		optimizee.symbolic_model.set_arch_paramters(alpha)

	def load_checkpoint(self, optimizee, filepath):
		"""
		Loads checkpoint saved by save_checkpoint()
		"""
		checkpoint = torch.load(filepath)
		start_epoch = checkpoint['epoch']
		start_episode = checkpoint['episode']
		optimizee.load_state_dict(checkpoint['optimizee'])
		optimizee.sync_symbolic_model()
		self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(filepath, checkpoint['epoch']))
		return start_epoch, start_episode

	def write_beta(self, beta):
		for i,b in enumerate(beta):
			beta_name = 'beta_{}'.format(i)
			self.tensorboard_writer.add_histogram(beta_name, b.data.cpu().numpy(), self.n_iter_scalar)
			if beta_name in self.betas_embedding:
				self.betas_embedding[beta_name]['data'] = torch.cat([self.betas_embedding[beta_name]['data'], b.data.cpu().unsqueeze(0)])
				self.betas_embedding[beta_name]['metadata'] = np.concatenate([self.betas_embedding[beta_name]['metadata'], np.array([self.n_iter_scalar])])
			else:
				self.betas_embedding[beta_name] = {}
				self.betas_embedding[beta_name]['data'] = b.data.cpu().unsqueeze(0)
				self.betas_embedding[beta_name]['metadata'] = np.array([self.n_iter_scalar])
			for dim in range(b.size(0)):
				self.tensorboard_writer.add_scalar('data/'+beta_name+'_dim_'+str(dim), b[dim].data.cpu(), self.n_iter_scalar)
			self.tensorboard_writer.add_scalar('data/'+beta_name+'_grad', sum(self.beta_grads[beta_name]) / len(self.beta_grads[beta_name]), self.n_iter_scalar)
		self.n_iter_scalar += 1
		# reset accum grad norms for each beta
		for k,v in self.beta_grads.items():
			self.beta_grads[k] = []

	def write_beta_embedding(self):
		for beta_name, emb in self.betas_embedding.items():
			self.tensorboard_writer.add_embedding(emb['data'], metadata=emb['metadata'], global_step=self.n_iter_embedding, tag=beta_name)
		self.n_iter_embedding += 1




