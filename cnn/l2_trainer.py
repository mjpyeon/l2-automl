import copy, math, genotypes, pdb
import os, sys
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Logger import logger
from meta_optimizer import MetaOptimizer
from network import *
from model_search import Darts_Network
from architect2 import Architect

from optimizee import Optimizee
from cifar10_dataset import Cifar10
from args import args
import pdb


class Saver:
	def save_checkpoint(self, optimizee, meta_optim, epoch, filepath):
		"""
		Save checkpoint of optimizee.model and meta_optim states including beta (and alpha if using darts arch)
		"""
		logger.log("=> saving checkpoint '{}'".format(filepath))
		state = {'epoch':int(epoch) + 1, 'optim_state':meta_optim.state_dict(), 'model_state':optimizee.model.state_dict()}
		torch.save(state, filepath)

	def save_beta(self, meta_optim, n_iter):
		if(not os.path.isdir(args.save_dir)):
			os.makedirs(args.save_dir)
		save_dir = args.save_dir + '/beta_' + str(n_iter)
		logger.log('=> save beta to {}'.format(save_dir))
		torch.save(meta_optim.state_dict(NoStates=True), save_dir)

	def load_beta(self, meta_optim, beta_states):
		"""
		Loads saved beta to given meta_optim
		"""
		meta_optim.load_state_dict(beta_states, NoStates=True)
		#logger.log("=> loaded beta/lr from '{}'".format(filepath))

	def load_alpha(self, optimizee, alpha):
		"""
		Loads saved alpha to given optimizee
		"""
		optimizee.model.set_arch_paramters(alpha)
		optimizee.symbolic_model.set_arch_paramters(alpha)

	def load_checkpoint(self, optimizee, meta_optim, filepath, scheduler):
		"""
		Loads checkpoint saved by save_checkpoint()
		"""
		checkpoint = torch.load(filepath)
		start_epoch = checkpoint['epoch']
		optimizee.model.load_state_dict(checkpoint['model_state'])
		optimizee.sync_symbolic_model()
		meta_optim.load_state_dict(checkpoint['optim_state'])
		scheduler.last_epoch = start_epoch - 1
		logger.log("=> loaded checkpoint '{}' (epoch {})".format(filepath, checkpoint['epoch']))
		return start_epoch


class Trainer:
	def __init__(self):
		torch.backends.cudnn.deterministic = True
		np.random.seed(args.seed)
		self.device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() else "cpu")
		torch.cuda.set_device(args.gpu)

		self.saver = Saver()
		self.dataset = Cifar10()
		self.criterion = nn.CrossEntropyLoss()

		if args.use_darts_arch:
			genotype = eval("genotypes.%s" % args.arch)
			self.Net = lambda: Darts_Network(args.init_channels, self.dataset.num_classes, args.layers, self.criterion)
			logger.log('Network DARTS')
		elif args.arch == 'VGG11':
			self.Net = lambda: VGG('VGG11')
		elif args.arch == 'mobilenet':
			self.Net = MobileNetV2
		elif args.arch == 'simple':
			self.Net = Simple_Net
		else:
			raise Exception('Net not defined!')

	def test_beta(self, beta_states):
		optimizee = Optimizee(self.Net().to(self.device))
		if args.use_darts_arch:
			architect = Architect(optimizee.model, args)
		else:
			architect = None
		meta_optim = MetaOptimizer(optimizee.model.parameters(), lr = args.learning_rate)
		meta_optim.load_state_dict(meta_beta_dict)
		meta_optim.beta_scaling = args.max_beta_scaling
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
									meta_optim, float(args.test_beta_epoch), eta_min=args.learning_rate_min)
		for epoch in range(args.test_beta_epoch):
			scheduler.step()
			lr = scheduler.get_lr()[0]
			if (epoch+1) % args.test_freq == 0:
				logger.log('Run Testing')
				acc = self.eval(optimizee.model)
				self.train_epoch(epoch, lr, optimizee, meta_optim, optimizer, architect)



	def train(self):
		meta_beta_dict, alpha, optimizer_dict = None, None, None
		best_tracker = {'beta_states':None, 'acc':0}
		for episode in range(args.max_episodes):
			torch.manual_seed(args.seed + episode)
			optimizee = Optimizee(self.Net().to(self.device))
			if args.use_darts_arch:
				architect = Architect(optimizee.model, args)
			else:
				architect = None
			torch.manual_seed(args.seed + episode)
			meta_optim = MetaOptimizer(optimizee.model.parameters(), lr = args.learning_rate)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
									meta_optim, float(args.epoch), eta_min=args.learning_rate_min)
			if(episode == 0): # at beginning, determine the model of training
				# check loading
				if(args.continue_checkpoint): # continue training from checkpoint
					assert os.path.exists(args.checkpoint_path), 'checkpoint_path not exists'
					start_epoch = self.saver.load_checkpoint(optimizee, meta_optim, args.checkpoint_path, scheduler)
				else:
					start_epoch = 0
					if(args.load_beta):
						self.saver.load_beta(meta_optim, beta_states = torch.load(args.checkpoint_path)['optim_state'])
					if(args.load_alpha):
						self.saver.load_alpha(optimizee, alpha = torch.load(args.checkpoint_path)['model_state']['arch_parameters'])

				# check if training
				if(args.train_beta):
					optimizer = optim.Adam(meta_optim.training_params(), lr = args.beta_learning_rate)	
					if(optimizer_dict is not None):
						optimizer.load_state_dict(optimizer_dict)
					logger.log_once('optimizer = optim.Adam')
				else:
					optimizer = None
			elif(type(meta_beta_dict) != type(None)): # after first episode, load beta (/alpha) from previous episode
				meta_optim.load_state_dict(meta_beta_dict)
				if(args.use_darts_arch):
					self.saver.load_alpha(optimizee, alpha = alpha)


			best_test_acc = 0
			for epoch in range(start_epoch, args.epoch):  # loop over the dataset multiple times
				scheduler.step()
				lr = scheduler.get_lr()[0]
				meta_optim.beta_scaling = np.interp(epoch + 1, [0, args.epoch], [args.min_beta_scaling, args.max_beta_scaling])
				drop_path_prob = args.drop_path_prob * epoch / float(args.epoch)
				optimizee.model.drop_path_prob, optimizee.symbolic_model.drop_path_prob = [drop_path_prob] * 2
				torch.manual_seed(args.seed + episode)
				self.train_epoch(epoch, lr, optimizee, meta_optim, optimizer, architect)
				if (epoch+1) % args.test_freq == 0:
					logger.log('Run Testing')
					acc = self.eval(optimizee.model)
					# if accuracy not above random, discard this example and start a new one
					if(acc <= 0.11):
						meta_beta_dict = None
						break
					if(args.save_dir != '' and best_test_acc < acc):
						best_test_acc = acc
						if(not os.path.isdir(args.save_dir)):
							os.makedirs(args.save_dir)
						checkpoint_path = args.save_dir+'/model_episode_{}_epoch_{}'.format(str(episode), str(epoch))
						self.saver.save_checkpoint(optimizee, meta_optim, epoch, checkpoint_path)
						logger.log('Saving to {}'.format(args.save_dir))
				if(args.train_beta):
					optimizer_dict = optimizer.state_dict()
				if(args.use_darts_arch):
					alpha = optimizee.model.arch_parameters()
			#meta_beta_dict = meta_optim.state_dict(NoStates=True)
			if(best_tracker['acc'] < acc):
				best_tracker['beta_states'] = meta_optim.state_dict(NoStates=True)
				best_tracker['acc'] = acc
		self.test_beta(best_tracker['beta_states'])


	def eval(self, model):
		correct, total = 0, 0
		with torch.no_grad():
			for images, labels in self.dataset.test_loader:
				images, labels = images.to(self.device), labels.to(self.device)
				if args.use_darts_arch and args.auxiliary:
					outputs, _ = model(images)
				else:
					outputs = model(images)
					if type(outputs) == type(()):
						outputs = outputs[0]
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		logger.log('Accuracy of the network on the 10000 test images: %d %%' % (
			100 * correct / total))
		return float(correct) / total

	def train_epoch(self, epoch, lr, optimizee, meta_optim, optimizer, architect):
		if(args.train_beta):
			self.train_epoch_(epoch, lr, optimizee, meta_optim, optimizer, architect)
			logger.write_beta_embedding()
			if(args.save_dir != ''):
				self.saver.save_beta(meta_optim, epoch)
			meta_optim.print_beta_greedy()
			logger.log('Finished Training')
			logger.log('beta[0] abs mean: {}'.format(meta_optim.beta[0].abs().mean()))
		else:
			self.train_epoch_fix_beta_(epoch, lr, optimizee, meta_optim, architect)
			meta_optim.print_beta_greedy()

	def train_epoch_fix_beta_(self, epoch, lr, optimizee, meta_optim, architect=None):
		losses, model_grad_norms= 0.0, 0.0
		for i, data in enumerate(self.dataset.train_loader, 0):
			x_train, y_train = data[0].to(self.device), data[1].to(self.device)
			val_data = next(iter(self.dataset.val_loader))
			x_val, y_val = val_data[0].to(self.device), val_data[1].to(self.device)
			# Derive \frac{\partial L}{\partial \theta}
			optimizee.model.zero_grad()
			loss = self.model_forward(optimizee.model, x_train, y_train)
			loss.backward()
			model_grad_norms += nn.utils.clip_grad_norm_(optimizee.model.parameters(), args.model_grad_norm)
			
			param_updates = meta_optim.step(DoUpdate=True)
			losses += loss.item()
			if args.use_darts_arch and args.train_alpha:
					arch_grad_norm = architect.step(x_train, y_train, x_val, y_val, lr, meta_optim, unrolled=args.unrolled)

			if (i+1) % args.log_freq == 0:    # print every 2000 mini-batches
				logger.log('[%d, %5d] loss: %.3f/%.3f' %
					  (epoch + 1, i + 1, losses / args.log_freq, model_grad_norms / args.log_freq))  
				if(args.use_darts_arch):
					logger.log('updates[-1] abs mean: {:.4f}, alpha[0] mean: {:.4f}, alpha grad norm: {:.4f}'
						.format(param_updates[-1].abs().mean(), optimizee.model.alphas_normal.abs().mean(), arch_grad_norm))
				else:
					logger.log('weight[-1] abs mean: {:.4f}. updates[-1] abs mean: {:.4f}'.format(optimizee.model.fc3._parameters['weight'].abs().mean(), param_updates[-1].abs().mean()))
				losses, model_grad_norms= 0.0, 0.0


	def train_epoch_(self, epoch, lr, optimizee, meta_optim, optimizer, architect=None):
		losses, next_step_losses, meta_update_losses = 0.0, 0.0, 0.0
		model_grad_norms, beta_grad_norms = 0., 0.
		for i, data in enumerate(self.dataset.train_loader, 0):
			x_train, y_train = data[0].to(self.device), data[1].to(self.device)
			val_data = next(iter(self.dataset.val_loader))
			x_val, y_val = val_data[0].to(self.device), val_data[1].to(self.device)
			# Derive \frac{\partial L}{\partial \theta}
			optimizee.model.zero_grad()
			loss = self.model_forward(optimizee.model, x_train, y_train)
			loss.backward()
			model_grad_norms += nn.utils.clip_grad_norm_(optimizee.model.parameters(), args.model_grad_norm)
			
			param_updates = meta_optim.step()

			# do a differentiable step of update over optimizee.symbolic_model
			optimizee.differentiable_update(param_updates)

			# check the next-step loss after update using the symbolic model
			# so that it is on the computational graph
			# choose to use d_train / d_val with preference to d_val when epoch increases
			logger.log_once('choose to use d_train / d_val with preference to d_val when epoch increases')
			if(np.random.rand() > (float(epoch + 1)/args.epoch)):
				next_step_x, next_step_y = x_train, y_train
			else:
				logger.log_once('first next_step as val_d')
				next_step_x, next_step_y = x_val, y_val
			next_step_loss = self.model_forward(optimizee.symbolic_model, x_train, y_train)
			if(math.isnan(next_step_loss.item())):
				raise Exception('next_step_loss is nan')
			
			next_step_losses += next_step_loss.item()
			losses += loss.item()
			
			meta_update_losses += next_step_loss

			logger.log_once('udpate if(new_loss.data < loss.data):')
			# do a non-differentiable step of update over optimizee.model if next_step_loss is smaller
			if(next_step_loss.data < loss.data):
				optimizee.update(param_updates)

			# forsee bptt_steps
			if((i+1) % args.bptt_step == 0):
				# compute grads for beta
				optimizer.zero_grad()
				meta_update_losses.backward()
				# normalize bptt gradient
				if args.normalize_bptt:
					for beta in meta_optim.beta:
						beta.grad /= args.bptt_step
					meta_optim.lr_scaling.grad /= args.bptt_step
				meta_optim.beta_entropy(args.beta_entropy_penalty)
				beta_grad_norms = nn.utils.clip_grad_norm_(meta_optim.training_params(), args.beta_grad_norm)
				with torch.no_grad():
					logger.add_beta_grads([(b.grad**2).mean() for b in meta_optim.beta])
				# update beta
				optimizer.step()
				# detach parameters 
				optimizee.detach()
				if args.use_darts_arch and args.train_alpha:
					arch_grad_norm = architect.step(x_train, y_train, x_val, y_val, lr, meta_optim, unrolled=args.unrolled)
				optimizee.sync_symbolic_model()
				meta_update_losses = 0
				
			if (i+1) % args.log_freq == 0:    # print every 2000 mini-batches
				logger.log('[%d, %5d] loss: %.3f/%.3f, next step loss: %.3f' %
					  (epoch + 1, i + 1, losses / args.log_freq, model_grad_norms / args.log_freq, next_step_losses/args.log_freq))  
				losses, next_step_losses = 0.0, 0.0
				if(args.use_darts_arch):
					logger.log('updates[-1] abs mean: {:.4f}, alpha[0] mean: {:.4f}, alpha grad norm: {:.4f}, beta[-1] data: {:.4f}, beta grad norm: {:.4f}, lr_scaling: {:.4f}'
						.format(param_updates[-1].abs().mean(), optimizee.model.alphas_normal.abs().mean(), arch_grad_norm,  meta_optim.beta[-1].abs().mean(), beta_grad_norms / args.log_freq, meta_optim.lr_scaling.item()))
				else:
					logger.log('weight[-1] abs mean: {:.4f}. updates[-1] abs mean: {:.4f}, beta[-1] data: {:.4f}, beta grad norm: {:.4f}, lr_scaling: {:.4f}'.format(optimizee.model.fc3._parameters['weight'].abs().mean(), param_updates[-1].abs().mean(), meta_optim.beta[-1].abs().mean(), beta_grad_norms / args.log_freq, meta_optim.lr_scaling.item()))
				logger.write_beta(meta_optim.cons, meta_optim.beta)
			#TODO: add some print statements for beta
			#if (i) % args.log_freq == 0:
			#    logger.log('beta[0] abs mean: {}, alpha[0] mean: {}'.format(meta_optim.beta[0].abs().mean(), model.alphas_normal.abs().mean()))
	
	def model_forward(self, model, inputs, labels):
		"""
		forward funtion for given model
		"""
		if args.use_darts_arch and args.auxiliary:
			outputs, logits_aux = model(inputs)
			loss = self.criterion(outputs, labels) + self.criterion(logits_aux, labels)
			del outputs, logits_aux
		else:
			outputs = model(inputs)
			if type(outputs) == type(()):
				outputs = outputs[0]
			loss = self.criterion(outputs, labels)
			del outputs
		return loss

if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()
