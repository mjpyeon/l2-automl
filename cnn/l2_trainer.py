import math
import pdb
import os
import sys
import datetime
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

import logger as L
from network import *
from auto_network import AutoNetwork
from optimizee import Optimizee
from cifar10_dataset import Cifar10
from args import args
from saver import Saver

np.random.seed(args.seed)
np.set_printoptions(precision=4)

class Trainer:
	def __init__(self):
		self._setup_logger()
		torch.backends.cudnn.deterministic = True
		self.device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() else "cpu")
		torch.cuda.set_device(args.gpu)
		self.dataset = Cifar10()
		self.saver = Saver(self.save_dir)
		if args.arch == 'auto':
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
		self.save_dir = os.path.join('./checkpoints', args.logdir)
		if not os.path.isdir(self.save_dir):
			os.makedirs(self.save_dir)
		log_path = os.path.join(self.save_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
		self.logger = L.create_logger(args.logdir, log_path)
		for arg in vars(args):
			self.logger.info("%-25s %-20s" % (arg, getattr(args, arg)))
	
	def test_beta(self, optimizee):
		self.logger.info('Testing beta')
		torch.manual_seed(2 * args.seed)
		optimizee.reset_model_parameters()
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizee.optimizer, 
							float(args.max_epoch), eta_min=args.learning_rate_min)
		optimizee.optimizer.print_beta_greedy()
		for epoch in range(0, args.max_test_epoch):
			scheduler.step()
			lr = scheduler.get_lr()[0]
			self.train_epoch(epoch, lr, optimizee, False)
			if (epoch + 1) % args.test_freq == 0:
				acc = self.eval(optimizee.model)

	def train(self):
		optimizee = Optimizee(self.Net)
		start_episode, start_epoch, best_test_acc = 0, 0, 0.
		if args.checkpoint != '':
			start_episode, start_epoch = self.saver.load_checkpoint(optimizee, args.checkpoint) 
		# Hao: in my implementation, the optimizers for alpha and beta will preserve their states across episodes
		for episode in range(start_episode, args.max_episodes):
			torch.manual_seed(args.seed + episode)
			optimizee.reset_model_parameters()
			if args.arch == 'auto':
				optimizee.reset_arch_parameters()
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizee.optimizer, 
							float(args.max_epoch), eta_min=args.learning_rate_min, last_epoch = start_epoch - 1)
			for epoch in range(start_epoch, args.max_epoch):  # loop over the dataset multiple times
				scheduler.step()
				lr = scheduler.get_lr()[0]
				training_status = self.train_epoch(epoch, lr, optimizee)
				if (epoch + 1) % args.test_freq == 0:
					acc = self.eval(optimizee.model)
					# if accuracy not above random, discard this example and start a new one
					if acc <= 0.11 or training_status == False:
						self.logger.warning('training_status false or acc too low, break')
						break
					checkpoint_path = self.save_dir + '/episode_{}_epoch_{}_acc_{}'.format(str(episode), str(epoch), str(acc))
					self.saver.save_checkpoint(optimizee, epoch, episode, checkpoint_path)
		return optimizee

	def train_epoch(self, epoch, lr, optimizee, train_beta=True):
		if train_beta:
			optimizee.optimizer.beta_temp = np.interp(epoch + 1, [0, args.max_epoch], [args.min_beta_temp, args.max_beta_temp])
			status = self._train_epoch(epoch, lr, optimizee)
			self.saver.write_beta_embedding()
			optimizee.optimizer.print_beta_greedy()
		else:
			optimizee.optimizer.beta_temp = args.max_beta_temp
			status = self._train_epoch_fix_beta(epoch, lr, optimizee)
		return status
	
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
				self.logger.error('next step loss becomes NaN, break')
				raise Exception
			next_step_losses += next_step_loss.item()
			losses += loss.item()
			meta_update_losses += next_step_loss
			
			# do a non-differentiable update over optimizee.model if next_step_loss is smaller
			if (next_step_loss.item() < loss.item()): # This is still important for performance
				optimizee.update(param_updates)

			# forsee bptt_steps then update beta
			if (i + 1) % args.bptt_step == 0:
				beta_grad_norms += optimizee.beta_step(meta_update_losses).item()
				# let saver save the grads for beta
				with torch.no_grad():
					self.saver.add_beta_grads([b.grad.abs().mean() for b in optimizee.optimizer.beta])
				meta_update_losses = 0
				optimizee.sync_symbolic_model()

			if args.arch_training and (i + 1) % args.update_alpha_step == 0 :
				optimizee.alpha_step(x_train, y_train, x_val, y_val, lr) # TODO: make sure alpha step won't change weights
				optimizee.sync_symbolic_model(skip_weights=True) 

			if (i + 1) % args.log_freq == 0:    # print every 2000 mini-batches
				beta_out = optimizee.optimizer.beta[-1].data.cpu().numpy()
				alpha_out = optimizee.model.alphas_normal[-1].data.cpu().numpy() if args.arch_training else ''
				self.logger.info('[%d, %5d] loss: %.3f/%.3f, next step loss: %.3f, beta[-1]/L2(g): %s/%s, alpha[-1]: %s' %
					  (epoch + 1, i + 1, losses / args.log_freq, model_grad_norms / args.log_freq, next_step_losses/args.log_freq, beta_out, beta_grad_norms / args.log_freq, alpha_out))
				losses, next_step_losses, model_grad_norms, beta_grad_norms = 0., 0., 0., 0.
				self.saver.write_beta(optimizee.optimizer.beta)
		return True

	def _train_epoch_fix_beta(self, epoch, lr, optimizee):
		losses, model_grad_norms= 0.0, 0.0
		for i, data in enumerate(self.dataset.train_loader, 0):
			x_train, y_train = data[0].to(self.device), data[1].to(self.device)
			val_data = next(iter(self.dataset.val_loader))
			x_val, y_val = val_data[0].to(self.device), val_data[1].to(self.device)
			optimizee.model.zero_grad()
			loss = optimizee.model.forward_pass(x_train, y_train)
			loss.backward()
			model_grad_norms += nn.utils.clip_grad_norm_(optimizee.model.parameters(), args.model_grad_norm)
			param_updates = optimizee.optimizer.step(do_update = True)
			losses += loss.item()
			if args.use_darts_arch and args.train_alpha:
				optimizee.alpha_step(x_train, y_train, x_val, y_val, lr)
			if (i + 1) % args.log_freq == 0:    # print every 2000 mini-batches
				beta_out = optimizee.optimizer.beta[-1].data.cpu().numpy()
				alpha_out = optimizee.model.alphas_normal[-1].data.cpu().numpy() if args.arch_training else 'None'
				self.logger.info('[%d, %5d] loss: %.3f/%.3f, beta[-1]: %s, alpha[-1]: %s' %
					  (epoch + 1, i + 1, losses / args.log_freq, model_grad_norms / args.log_freq, beta_out, alpha_out))  
				losses, model_grad_norms= 0., 0.
		return True

if __name__ == '__main__':
	trainer = Trainer()
	optimizee = trainer.train()
	trainer.test_beta(optimizee)
