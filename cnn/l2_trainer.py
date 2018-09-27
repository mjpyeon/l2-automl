import math
import pdb
import os
import sys
import datetime
import numpy as np
from collections import deque
class List(deque):
    def mean(self):
        return sum(self) / len(self) + 1e-12

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
		self.set_seed(args.seed)
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

	def set_seed(self, seed):
		torch.manual_seed(seed)
		np.random.seed(seed)

	def _setup_logger(self):
		self.save_dir = os.path.join('./checkpoints', args.logdir)
		if not os.path.isdir(self.save_dir):
			os.makedirs(self.save_dir)
		log_path = os.path.join(self.save_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
		self.logger = L.create_logger(args.logdir, log_path)
		self.logger.info(args.message)
		for arg in vars(args):
			self.logger.info("%-25s %-20s" % (arg, getattr(args, arg)))
	
	def test_beta(self, optimizee):
		self.logger.info('Testing beta')
		self.set_seed(2 * args.seed)
		optimizee.reset_model_parameters()
		optimizee.reset_optimizer_state()
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizee.optimizer, 
							float(args.max_epoch), eta_min=args.learning_rate_min)
		optimizee.optimizer.print_beta_greedy()
		optimizee.optimizer.beta_inv_temp = args.max_beta_inv_temp
		choices = optimizee.optimizer.greedy_sample_choices()
		optimizee.optimizer.print_choices(choices)
		for epoch in range(0, args.max_test_epoch):
			scheduler.step()
			lr = scheduler.get_lr()[0]
			self.train_epoch(epoch, lr, optimizee, False, choices=choices)
			if (epoch + 1) % args.test_freq == 0:
				acc = self.eval(optimizee.model)

	def train(self):
		self.set_seed(args.seed)
		optimizee = Optimizee(self.Net)
		start_episode, start_epoch, best_test_acc = 0, 0, 0.
		if args.checkpoint != '':
			start_episode, start_epoch = self.saver.load_checkpoint(optimizee, args.checkpoint) 
		# Hao: in my implementation, the optimizers for alpha and beta will preserve their states across episodes
		for episode in range(start_episode, args.max_episodes):
			self.set_seed(args.seed + episode)
			optimizee.reset_model_parameters()
			optimizee.reset_optimizer_state()
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
					if acc <= 0.15 or training_status == False:
						self.logger.warning('training_status false or acc too low, break')
						break
					checkpoint_path = self.save_dir + '/episode_{}_epoch_{}_acc_{}'.format(str(episode), str(epoch), str(acc))
					self.saver.save_checkpoint(optimizee, epoch, episode, checkpoint_path)
		return optimizee

	def train_epoch(self, epoch, lr, optimizee, train_beta=True, choices=None):
		if train_beta:
			optimizee.optimizer.beta_inv_temp = np.interp(epoch + 1, [0, args.max_epoch], [args.min_beta_inv_temp, args.max_beta_inv_temp])
			status = self._train_epoch(epoch, lr, optimizee)
			self.saver.write_beta_embedding()
			optimizee.optimizer.print_beta_greedy()
		else:
			status = self._train_epoch_fix_beta(epoch, lr, optimizee, choices)
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

	def sample_choices(self, optimizee):
		with torch.no_grad():
			for i in range(10):
				choices = optimizee.optimizer.gumble_sample_choices()
				optimizee.optimizer.print_choices(choices)

	def _train_epoch(self, epoch, lr, optimizee):
		meta_update_losses = 0.0
		losses, next_step_losses = List(maxlen=args.log_freq), List(maxlen=args.log_freq)
		model_grad_norms, beta_grad_norms = 0., 0.
		'''
		if(epoch == 0):#first epoch, using sgd to normalize momentum, v, r, etc
			optimizee.optimizer.param_groups[0]['lr'] = 0.1
			optimizee.optimizer.init_beta_gold()
		if(epoch == 1):
			optimizee.optimizer.param_groups[0]['lr'] = args.learning_rate
			optimizee.optimizer.init_beta()
		'''
		choices = optimizee.optimizer.gumble_sample_choices()
		
		for i, data in enumerate(self.dataset.train_loader, 0):
			x_train, y_train = data[0].to(self.device), data[1].to(self.device)
			val_data = next(iter(self.dataset.val_loader))
			x_val, y_val = val_data[0].to(self.device), val_data[1].to(self.device)

			# Derive \frac{\partial L}{\partial \theta}
			optimizee.model.zero_grad()
			loss = optimizee.model.forward_pass(x_train, y_train)
			loss.backward()
			model_grad_norms += nn.utils.clip_grad_norm_(optimizee.model.parameters(), args.model_grad_norm)
			if math.isnan(model_grad_norms.item()) or model_grad_norms.item() > 1e8:
				self.logger.warning('model_grad_norms nan')
				#pdb.set_trace()
			optimizee.optimizer.gumbel_temp = np.maximum(args.max_gumbel_temp*np.exp(-args.gumbel_anneal_rate*i),args.min_gumbel_temp)
			#if epoch < 10:
			#	param_updates, choices = optimizee.optimizer.step(use_gumble=False)
			#else:
			param_updates, choices, params_grad = optimizee.optimizer.step(choices=choices)

			# do a differentiable step of update over optimizee.symbolic_model
			optimizee.differentiable_update(param_updates)

			# check the next-step loss after update using the symbolic model
			# so that it is on the computational graph
			# TODO use validation data? not sure
			next_step_loss = optimizee.symbolic_model.forward_pass(x_train, y_train)
			if(math.isnan(next_step_loss.item()) or next_step_loss.item() > 1e10 or next_step_loss.item() == 0):
				next_step_loss = sum([choice.max() for choice in choices]) * 1e2 # use choices.sum() as proxy to bp

			next_step_losses += [next_step_loss.item()]
			losses += [loss.item()]
			if loss.item() < 0.01:
				pdb.set_trace()
			meta_update_losses += sum([choice.max() for choice in choices]) * (next_step_loss.item() - losses.mean())#next_step_loss#
			# do a non-differentiable update over optimizee.model if next_step_loss is smaller or at increasing probability
			if (next_step_loss.item() < loss.item() 
				or (np.random.rand() < float(epoch-10)/args.max_epoch and next_step_loss.item() < 1.5*loss.item()) ): 
				optimizee.update(param_updates, params_grad)

			#if i == 0:
			#	pdb.set_trace()
			#	pass
			# forsee bptt_steps then update beta
			if (i + 1) % args.bptt_step == 0:
				beta_grad_norms += optimizee.beta_step(meta_update_losses).item()
				# let saver save the grads for bete
				with torch.no_grad():
					self.saver.add_beta_grads([b.grad.abs().mean() for b in optimizee.optimizer.beta])
				meta_update_losses = 0
				optimizee.sync_symbolic_model()
				choices = optimizee.optimizer.gumble_sample_choices()
				#if(epoch >= 1):
				#optimizee.optimizer.print_choices(choices)

			if args.arch_training and (i + 1) % args.update_alpha_step == 0 :
				optimizee.alpha_step(x_train, y_train, x_val, y_val, lr) # TODO: make sure alpha step won't change weights
				optimizee.sync_symbolic_model(skip_weights=True) 

			if epoch >= 0 and (i + 1) % args.log_freq == 0:    # print every 2000 mini-batches
				beta_out = optimizee.optimizer.beta[-1].data.cpu().numpy()
				alpha_out = optimizee.model.alphas_normal[-1].data.cpu().numpy() if args.arch_training else ''
				self.logger.info('[%d, %5d] loss: %8.2e/%7.1e, next step loss: %8.2e, beta[-1]/L2(g): %s/%7.1e, alpha[-1]: %s' %
					  (epoch + 1, i + 1, losses.mean(), model_grad_norms / args.log_freq, next_step_losses.mean(), beta_out, beta_grad_norms / args.log_freq, alpha_out))
				optimizee.optimizer.print_beta_greedy()
				optimizee.optimizer.print_choices(choices)
				model_grad_norms, beta_grad_norms = 0., 0.
				self.saver.write_beta(optimizee.optimizer.beta)
		return True

	def _train_epoch_fix_beta(self, epoch, lr, optimizee, choices):
		losses, model_grad_norms= 0.0, 0.0
		for i, data in enumerate(self.dataset.train_loader, 0):
			x_train, y_train = data[0].to(self.device), data[1].to(self.device)
			val_data = next(iter(self.dataset.val_loader))
			x_val, y_val = val_data[0].to(self.device), val_data[1].to(self.device)
			optimizee.model.zero_grad()
			loss = optimizee.model.forward_pass(x_train, y_train)
			loss.backward()
			model_grad_norms += nn.utils.clip_grad_norm_(optimizee.model.parameters(), args.model_grad_norm)
			_, _, _ = optimizee.optimizer.step(virtual=False, do_update = True, choices = choices)
			losses += loss.item()
			if args.use_darts_arch and args.train_alpha:
				optimizee.alpha_step(x_train, y_train, x_val, y_val, lr)
			if (i + 1) % max(args.log_freq, 50) == 0:    # print every 2000 mini-batches
				log_freq = max(args.log_freq, 50)
				beta_out = optimizee.optimizer.beta[-1].data.cpu().numpy()
				alpha_out = optimizee.model.alphas_normal[-1].data.cpu().numpy() if args.arch_training else 'None'
				self.logger.info('[%d, %5d] loss: %.3f/%.3f, beta[-1]: %s, alpha[-1]: %s' %
					  (epoch + 1, i + 1, losses / log_freq, model_grad_norms / log_freq, beta_out, alpha_out))  
				losses, model_grad_norms= 0., 0.
		return True

if __name__ == '__main__':
	trainer = Trainer()
	optimizee = trainer.train()
	trainer.test_beta(optimizee)
