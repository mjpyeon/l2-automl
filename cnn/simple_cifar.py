import copy, math, genotypes, pdb
import numpy as np
from collections import deque

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from meta_optimizer import MetaOptimizer
from model_search import Network
from architect2 import Architect

from network import *
from optimizee import Optimizee
from cifar10_dataset import Cifar10
from args import args

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

if args.use_darts_arch:
    genotype = eval("genotypes.%s" % args.arch)
    Net = lambda: Network(args.init_channels, dataset.num_classes, args.layers, criterion)
    print('Network DARTS')
elif args.arch is 'VGG11':
    Net = lambda: VGG('VGG11')
elif args.arch is 'mobilenet':
    Net = MobileNetV2
    pass

dataset = Cifar10()
criterion = nn.CrossEntropyLoss()
old_beta = None

for episode  in range(args.max_episodes):
	optimizee = Optimizee(Net().to(device))
	if args.use_darts_arch:
	    architect = Architect(model, args)
	meta_optim = MetaOptimizer(optimizee.model.parameters(), lr = args.learning_rate, beta = old_beta)
	optimizer = optim.Adam(meta_optim.beta + [meta_optim.lr], lr = args.beta_learning_rate, 
	                       weight_decay = args.beta_weight_decay)
	
	next_step_loss_queue = deque(maxlen = 100) 
	next_step_loss_queue.append(2.2) # 2.2 is the baseline per observation
	for epoch in range(args.epoch):  # loop over the dataset multiple times
		losses, next_step_losses, meta_update_loss = 0.0, 0.0, 0.0
		for i, data in enumerate(dataset.train_loader, 0):
			x_train, y_train = data[0].to(device), data[1].to(device)
			val_data = next(iter(dataset.val_loader))
			x_val, y_val = val_data[0].to(device), val_data[1].to(device)
			# Derive \frac{\partial L}{\partial \theta}
			optimizee.model.zero_grad()
			if args.use_darts_arch and args.auxiliary:
			    outputs, logits_aux = optimizee.model(x_train)
			    loss = criterion(outputs, y_train) + criterion(logits_aux, y_train)
			else:
				outputs = optimizee.model(x_train)
				loss = criterion(outputs, y_train)
			loss.backward()
			nn.utils.clip_grad_norm_(optimizee.model.parameters(), args.model_grad_norm)
			
			param_updates = meta_optim.step()
			# do a non-differentiable step of update over optimizee.model
			optimizee.update(param_updates)
			# do a differentiable step of update over optimizee.symbolic_model
			optimizee.differentiable_update(param_updates)

			# check the next-step loss after update using the symbolic model
			# so that it is on the computational graph
			# TODO use validation data? not sure
			if args.use_darts_arch and args.auxiliary:
			    outputs, logits_aux = optimizee.symbolic_model(x_val)
			    next_step_loss = criterion(outputs, y_val) + criterion(logits_aux, y_val)
			else:
				outputs = optimizee.symbolic_model(x_val)
				next_step_loss = criterion(outputs, y_val)
			if(math.isnan(next_step_loss.item())):
			    raise Exception('next_step_loss is nan')
			
			next_step_loss_queue.append(next_step_loss.item())
			next_step_losses += next_step_loss.item()
			losses += loss.item()
			
			meta_update_loss = meta_update_loss + next_step_loss

			# forsee bptt_steps
			print('iteration {}'.format(i))
			if((i+1) % args.bptt_step == 0):
			    # compute grads for beta
				optimizer.zero_grad()
				meta_update_loss.backward()
				# normalize bptt gradient
				if args.normalize_bptt:
					for beta in meta_optim.beta:
						beta.grad /= args.bptt_step
					meta_optim.lr.grad /= args.bptt_step
				grad_norm = nn.utils.clip_grad_norm_(meta_optim.beta, args.beta_grad_norm)
				# update beta
				optimizer.step()
				# detach parameters 
				optimizee.detach()
				meta_update_loss = 0
				#if args.use_darts_arch and args.arch_training:
				#    arch_grad_norm = architect.step(inputs, labels, x_val, y_val, meta_optim.lr.data.item(), meta_optim, unrolled=args.unrolled)
			if (i+1) % args.log_freq == 0:    # print every 2000 mini-batches
			    print('[%d, %5d] loss: %.3f, new loss: %.3f' %
			          (epoch + 1, i + 1, losses / args.log_freq, next_step_losses/args.log_freq))
			    running_losses = 0.0
			    running_new_losses = 0.0
			#TODO: add some print statements for beta
			#if (i) % args.log_freq == 0:
			#    print('beta[0] abs mean: {}, alpha[0] mean: {}'.format(meta_optim.beta[0].abs().mean(), model.alphas_normal.abs().mean()))
		meta_optim.print_beta_code()
		if (epoch+1) % args.test_freq == 0:
			print('Run Testing')
			acc = dataset.eval(optimizee.model)
			# if accuracy not above random, discard this example and start a new one
			if(acc <= 0.11):
			    break
			if args.save_path != '':
			    torch.save(model, args.save_path+'/model_iter'+str(iteration))
			    torch.save(meta_optim, args.save_path+'/meta_optim_iter'+str(iteration))
			    print('Saving to {}'.format(args.save_path))
	old_beta = meta_optim.beta
	print('Finished Training')
	print('beta[0] abs mean: {}'.format(meta_optim.beta[0].abs().mean()))
	dataset.eval(model)
