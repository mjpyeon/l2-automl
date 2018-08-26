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

import genotypes
import logger as L
from network import *
from auto_network import AutoNetwork
from optimizee import Optimizee
from cifar10_dataset import Cifar10
from args import args

np.random.seed(args.seed)

class Trainer:
	def __init__(self):
		self._setup_logger()
		torch.backends.cudnn.deterministic = True
		self.device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() else "cpu")
		torch.cuda.set_device(args.gpu)
		self.dataset = Cifar10()
		if args.arch == 'auto':
			genotype = eval("genotypes.%s" % args.arch)
			self.Net = lambda: AutoNetwork(args.init_channels, self.dataset.num_classes, args.layers, nn.CrossEntropyLoss())
		elif args.arch == 'VGG11':
			self.Net = lambda: VGG('VGG11', self.dataset.num_classes, nn.CrossEntropyLoss())
		elif args.arch == 'mobilenet':
			self.Net = lambda: MobileNetV2(self.dataset.num_classes, nn.CrossEntropyLoss())
		elif args.arch == 'simple':
			self.Net = lambda: SimpleNetwork(self.dataset.num_classes, nn.CrossEntropyLoss())
		else:
			raise Exception('Net not defined!')

	def _setup_logger(self):
		save_dir = os.path.join('./checkpoints', args.logdir)
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		log_path = os.path.join(save_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
		self.logger = L.create_logger(args.logdir, log_path)
		for arg in vars(args):
			self.logger.info("%-25s %-20s" % (arg, getattr(args, arg)))

	def train(self):
		optimizee = Optimizee(self.Net)
		start_episode, start_epoch, best_test_acc = 0, 0, 0.0
		if args.checkpoint != '':
			start_episode, start_epoch, optimizee = self.load_checkpoint(args.checkpoint, optimizee) 
		# Hao: in my implementation, the optimizers for alpha and beta will preserve their states across episodes
		for episode in range(start_episode, args.max_episodes):
			torch.manual_seed(args.seed + episode)
			if episode > 0:
				optimizee.reset_arch_parameters()
				optimizee.reset_model_parameters()
			#TODO(Hailin) it seems this lr is only used to optimize alpha
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizee.optimizer, 
							float(args.max_epoch), eta_min=args.learning_rate_min, last_epoch = start_epoch - 1)
			for epoch in range(start_epoch, args.max_epoch):  # loop over the dataset multiple times
				scheduler.step()
				lr = scheduler.get_lr()[0]
				#TODO (Hailin): I don't see any usage of drop_path_prob, so I remove it -- please double check
				self._train_epoch(epoch, lr, optimizee)
				optimzee.optimizer.print_beta_greedy()
				print('beta[0] abs mean: {}'.format(optimizer.beta[0].abs().mean()))
				if (epoch + 1) % args.test_freq == 0:
					acc = self.eval(optimizee.model)
					# if accuracy not above random, discard this example and start a new one
					if acc <= 0.11:
						break
					checkpoint_path = args.save_dir + '/episode_{}_epoch_{}'.format(str(episode), str(epoch))
					self.save_checkpoint(checkpoint_path, episode, epoch, optimizee)
					if best_test_acc < acc:
						best_test_acc = acc
						self.save_best_checkpoint(episode, epoch, optimizee)

	def eval(self, model):
		correct, total = 0, 0
		with torch.no_grad():
			for images, labels in self.dataset.test_loader:
				images, labels = images.to(self.device), labels.to(self.device)
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		self.logger.info('Accuracy of the network on the 10000 test images: %d %%' % (
			100 * correct / total))
		return float(correct) / total

	def _train_epoch(self, epoch, lr, optimizee):
		losses, next_step_losses, meta_update_losses = 0.0, 0.0, 0.0
		model_grad_norms, beta_grad_norms = 0., 0.
		optimizee.optimizer.beta_temp = np.interp(epoch + 1, [0, args.max_epoch], [args.min_beta_temp, args.max_beta_temp])
		for i, data in enumerate(self.dataset.train_loader, 0):
			x_train, y_train = data[0].to(self.device), data[1].to(self.device)
			val_data = next(iter(self.dataset.val_loader))
			x_val, y_val = val_data[0].to(self.device), val_data[1].to(self.device)

			# Derive \frac{\partial L}{\partial \theta}
			optimizee.model.zero_grad()
			loss = optimizee.model.forward_pass(x_train, y_train)
			loss.backward()
			model_grad_norms += nn.utils.clip_grad_norm_(optimizee.model.parameters(), args.model_grad_norm)
			param_updates = optimizee.optimizer.step()

			# do a differentiable step of update over optimizee.symbolic_model
			optimizee.differentiable_update(param_updates)

			# check the next-step loss after update using the symbolic model
			# so that it is on the computational graph
			# TODO use validation data? not sure
			next_step_loss = optimizee.symbolic_model.forward_pass(x_train, y_train)
			if(math.isnan(next_step_loss.item())):
				error_msg = 'next step loss becomes NaN'
				self.logger.error(error_msg)
				raise Exception(error_msg)
			next_step_losses += next_step_loss.item()
			losses += loss.item()
			meta_update_losses += next_step_loss

			# do a non-differentiable update over optimizee.model if next_step_loss is smaller
			#if (next_step_loss.data < loss.data):
			optimizee.update(param_updates)

			# forsee bptt_steps then update beta
			if (i + 1) % args.bptt_step == 0:
				optimizee.beta_step(meta_update_losses)
				meta_update_losses = 0
				optimizee.sync_symbolic_model()

			if args.arch_training and (i + 1) % args.update_alpha_step == 0 :
				optimizee.alpha_step(x_train, y_train, x_val, y_val, lr)
				optimizee.sync_symbolic_model()

			if (i + 1) % args.log_freq == 0:    # print every 2000 mini-batches
				self.logger.info('[%d, %5d] loss: %.3f/%.3f, next step loss: %.3f' %
					  (epoch + 1, i + 1, losses / args.log_freq, model_grad_norms / args.log_freq, next_step_losses/args.log_freq))  
				losses, next_step_losses, model_grad_norms = 0.0, 0.0, 0.0
				#if(args.use_darts_arch):
				#	self.logger.info('updates[-1] abs mean: {:.4f}, alpha[0] mean: {:.4f}, alpha grad norm: {:.4f}, beta[-1] data: {:.4f}, beta grad norm: {:.4f}, lr_scaling: {:.4f}'
				#		.format(param_updates[-1].abs().mean(), optimizee.model.alphas_normal.abs().mean(), arch_grad_norm,  optimizee.optimizer.beta[-1].abs().mean(), beta_grad_norms / args.log_freq, optim.lr_scaling.item()))
				#else:
				#	self.logger.info('weight[-1] abs mean: {:.4f}. updates[-1] abs mean: {:.4f}, beta[-1] data: {:.4f}, beta grad norm: {:.4f}, lr_scaling: {:.4f}'.format(optimizee.model.fc3._parameters['weight'].abs().mean(), param_updates[-1].abs().mean(), meta_optim.beta[-1].abs().mean(), beta_grad_norms / args.log_freq, meta_optim.lr_scaling.item()))
			#TODO: add some print statements for beta
			#if (i) % args.log_freq == 0:
			#    print('beta[0] abs mean: {}, alpha[0] mean: {}'.format(meta_optim.beta[0].abs().mean(), model.alphas_normal.abs().mean()))

	def save_checkpoint(self, path, episode, epoch, optimizee):
		self.logger.info("=> saving checkpoint '{}'".format(path))
		state = {'episode': episode, 'epoch': epoch, 'optimzee': optimizee.state_dict()}
		torch.save(state, path)
	
	def save_best_checkpoint(self, episode, epoch, optimizee):
		path = args.save_dir + '/best.ckpt'
		self.save_checkpoint(path, episode, epoch, optimizee)

	def load_checkpoint(self, checkpoint_path, optimizee):
		"""
		Loads checkpoint saved by save_checkpoint()
		"""
		checkpoint = torch.load(checkpoint_path)
		episode = checkpoint['episode']
		epoch = checkpoint['epoch']
		optimizee.load_state_dict(checkpoint['optimizee'])
		self.logger.info("=> loaded checkpoint '{}' (episode {}, epoch {})".format(filepath, episode, epoch))
		return episode, epoch, optimizee 

if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()
