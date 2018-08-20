import copy, math, genotypes, pdb
import os, sys
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from args import args
from datetime import datetime


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(object):
	__metaclass__ = Singleton
	def __init__(self):
		current_time = datetime.now().strftime('%b%d_%H-%M-%S')
		self.tensorboard_writer = SummaryWriter(log_dir=args.log_dir+'/tensorboard_'+current_time)
		self.logged_message = set()
		self.betas_embedding = {}
		self.beta_grads = {}
		self.file = open(args.log_dir+'/'+current_time+'_log.txt', 'a+')
		self.n_iter_beta = 0
		self.print_args()
	
	def __exit__():
		self.file.close()

	def print_args(self):
	    for arg in vars(args):
	        self.log("%-25s %-20s" % (arg, getattr(args, arg)))


	def log(self, message):
		print(message)
		self.file.write(message+'\n')

	def log_once(self, message):
		"""
		Print function that only prints if message not printed before
		"""
		mhash = hash(message)
		if mhash in self.logged_message:
			return
		else:
			print(message)
			self.logged_message.add(mhash)
		self.file.write(message+'\n')

	def add_beta_grads(self, grads):
		for i,g in enumerate(grads):
			beta_name = 'beta_{}'.format(i)
			if(beta_name not in self.beta_grads):
				self.beta_grads[beta_name] = []
			self.beta_grads[beta_name] += [g.data.cpu().numpy()]


	def write_beta(self, cons, beta):
		for i,b in enumerate(beta):
			beta_name = 'beta_{}'.format(i)
			self.tensorboard_writer.add_histogram(beta_name, torch.cat([cons, b]).data.cpu().numpy(), self.n_iter_beta)
			if beta_name in self.betas_embedding:
				self.betas_embedding[beta_name]['data'] = torch.cat([self.betas_embedding[beta_name]['data'], b.data.cpu().unsqueeze(0)])
				self.betas_embedding[beta_name]['metadata'] = np.concatenate([self.betas_embedding[beta_name]['metadata'], np.array([self.n_iter_beta])])
			else:
				self.betas_embedding[beta_name] = {}
				self.betas_embedding[beta_name]['data'] = b.data.cpu().unsqueeze(0)
				self.betas_embedding[beta_name]['metadata'] = np.array([self.n_iter_beta])
			for dim in range(b.size(0)):
				self.tensorboard_writer.add_scalar('data/'+beta_name+'_dim_'+str(dim), b[dim].data.cpu(), self.n_iter_beta)
			self.tensorboard_writer.add_scalar('data/'+beta_name+'_grad', sum(self.beta_grads[beta_name]) / len(self.beta_grads[beta_name]), self.n_iter_beta)
		self.n_iter_beta += 1
		# reset accum grad norms for each beta
		for k,v in self.beta_grads.items():
			self.beta_grads[k] = []

	def write_beta_embedding(self):
		for beta_name, emb in self.betas_embedding.items():
			self.tensorboard_writer.add_embedding(emb['data'], metadata=emb['metadata'], global_step=len(emb['metadata']), tag=beta_name)

logger = Logger()