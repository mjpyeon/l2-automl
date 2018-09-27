import copy
import numpy as np
from operator import mul

import torch
import torchvision
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

from auto_optimizer import AutoOptimizer
from args import args
import logger as L
import math
import pdb

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

"""
An optimizee owns a meta optimier and a meta network, both learnable
In Optimizee class, we keep two copies of the optimizee model:
`model`: the model that can be BPed by L and updated by meta optimizer
`symbolic_model`: the model that is used as a variable in the computational graph, and BPed by L'. Its parameters,
				  however, can only be updated by copying from model (below the level of computational graph)
"""
class Optimizee:
	def __init__(self, Net):
		self.Net = Net
		self.logger = L.get_logger(args.logdir)
		self.model = self.Net().cuda()
		if args.arch == 'auto' and args.arch_training:
			self.alpha_optimizer = torch.optim.Adam(self.model.arch_parameters(),
							lr=args.alpha_learning_rate, betas=(0.5, 0.999), weight_decay=args.alpha_weight_decay)
		# create the auto learned optimizer
		self.symbolic_model = copy.deepcopy(self.model)
		self.optimizer = AutoOptimizer(self.model.parameters(), lr = args.learning_rate)
		self.beta_optimizer = torch.optim.SGD(self.optimizer.optim_parameters(), lr = args.beta_learning_rate, momentum=0.9)
		#self.traditional_optimizer = torch.optim.RMSprop(self.model.parameters(), lr = args.learning_rate)
		self.logger.info('beta optimizer: {}'.format(type(self.beta_optimizer)))

	def sync_symbolic_model(self, skip_weights=False):
		if skip_weights == False:
			self.symbolic_model.copy_params_from(self.model)
		if args.arch == 'auto':
			self.symbolic_model.set_arch_paramters(self.model.arch_parameters)

	def update(self, updates, params_grad):
		"""
		non-differentiable update
		"""
		for param, update in zip(self.model.parameters(), updates):
			param.data.add_(update.data)
		#self.optimizer.update_states(params_grad)
	
	def differentiable_update(self, updates):
		"""
		differentiable update, i.e. we assign a differentiable variable as the model parameters of a NN, which
		is disallowed by PyTorch actually. The way we do this is to first create a new variable `updated_param` 
		to store the updated result and then assign it to the model
		There seems to be no better way to reassign the parameter as a new variable than directly accessing _parameters
		"""
		self.count = 0
		def _iter_child(module):
			for child in module.children():
				if len(child._parameters) > 0:
					for key, param in child._parameters.items():
						if param is not None:
							updated_params = child._parameters[key] + updates[self.count]
							child._parameters[key] = updated_params 
							self.count += 1
				elif len(child._modules) > 0:
					_iter_child(child)
		_iter_child(self.symbolic_model)

	def detach(self):
		"""
		This detaches the symbolic_model.parameters from the past path that generates it,
		in order to perform truncated BPTT within every T steps.
		i.e. this will be called every T steps 
		"""
		def _iter_child(module):
			for child in module.children():
				if len(child._parameters) > 0:
					for key, param in child._parameters.items():
						if param is not None:
							child._parameters[key] = param.detach()
				elif len(child._modules) > 0:
					_iter_child(child)
		_iter_child(self.symbolic_model)


	def beta_step(self, meta_update_loss):
		self.beta_optimizer.zero_grad()
		meta_update_loss.backward()
		
		for beta in self.optimizer.beta:
			if beta.grad is None:
				beta.grad = torch.zeros_like(beta)

		# check if grad is nan
		for beta in self.optimizer.beta:
			if math.isnan(beta.grad.mean().item()):
				self.detach()
				return torch.Tensor([float('nan')])

		#normalize bptt grad
		if args.normalize_bptt:
			for beta in self.optimizer.beta:
				beta.grad /= args.bptt_step
			#if self.optimizer.lr_scaling.grad is not None:
			#	self.optimizer.lr_scaling.grad /= args.bptt_step

		# add entropy penalty and derive respective grads
		# self.optimizer.beta_entropy(args.beta_entropy_penalty)
		# grad normalization
		beta_grad_norms = nn.utils.clip_grad_norm_(self.optimizer.beta, args.beta_grad_norm)
		# perform an update on beta
		self.beta_optimizer.step()
		# then detach the params of symbolic_model from its past
		self.detach()
		# return beta grad norm
		return beta_grad_norms

	def alpha_step(self, x_train, y_train, x_val, y_val, eta):
		'''
		Perform one step of update of the architecture parameters alpha
		unrolled: if True, second order approximation (foresee one step ahead), else not
		eta: learning rate
		'''
		self.alpha_optimizer.zero_grad()
		if args.unrolled:
			self._backward_step_unrolled(x_train, y_train, x_val, y_val, eta)
		else:
			self._backward_step(x_val, y_val)
		grad_norm = nn.utils.clip_grad_norm_(self.model.arch_parameters(), 10.) #TODO(Hailin): why 10?
		self.alpha_optimizer.step()
		return grad_norm

	def _compute_unrolled_model(self, x, y, eta):
		loss = self.model._loss(x, y)
		loss.backward()
		theta = _concat(self.model.parameters()).data
		# the first part has been negated in meta optimizer
		dtheta = _concat(self.optimizer.step(virtual=True)).data - args.weight_decay * theta # l2 might be unnecessary
		unrolled_model = self._construct_model_from_theta(theta.add(eta, dtheta))
	
	def _backward_step(self, x, y):
		'''
		Generate grad_alpha given x, y and the current state of theta (at t)
		'''
		loss = self.model._loss(x, y)
		loss.backward()

	def _backward_step_unrolled(self, x_train, y_train, x_val, y_val, eta):
		unrolled_model = self._compute_unrolled_model(x_train, y_train, eta) # use training data to foresee a step
		unrolled_loss = unrolled_model._loss(x_val, y_val)
		unrolled_loss.backward()
		
		dalpha = [v.grad for v in unrolled_model.arch_parameters()]
		vector = [v.grad.data for v in unrolled_model.parameters()]  # TODO(Hao): a bit confused here - to check with Darts paper
		implicit_grads = self._hessian_vector_product(vector, x_train, y_train)
		
		for g, ig in zip(dalpha, implicit_grads):
			g.data.sub_(eta, ig.data)
		
		for v, g in zip(self.model.arch_parameters(), dalpha):
			if v.grad is None:
				v.grad = Variable(g.data)
			else:
				v.grad.data.copy_(g.data)

	def _update_model_from_theta(self, flat_params):
		# Restore original shapes
		self.offset = 0
		def _iter_child(parent_module):
			for i, module in enumerate(parent_module.children()):
				# if has sub modules
				if(len(module._modules) > 0):
					_iter_child(module)
				elif len(module._parameters) > 0:
					for key,param in module._parameters.items():
						if(type(param) != type(None)):
							flat_size = reduce(mul, param.size(), 1)
							module._parameters[key] = flat_params[self.offset:self.offset + flat_size].view(param.size())
							self.offset += flat_size
		_iter_child(self.model)

	def _constrcut_model_from_theta(self, theta):
		model_new = self.model.new()
		model_dict = self.model.state_dict()
		
		params, offset = {}, 0
		for k, v in self.model.named_parameters():
			v_length = np.prod(v.size())
			params[k] = theta[offset: offset+v_length].view(v.size())
			offset += v_length
		assert offset == len(theta)
		model_dict.update(params)
		model_new.load_state_dict(model_dict)
		return model_new.cuda()

	def _hessian_vector_product(self, vector, input, target, r=1e-2):
		R = r / _concat(vector).norm()
		for p, v in zip(self.model.parameters(), vector):
			p.data.add_(R, v)
		loss = self.model._loss(input, target)
		grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
		
		for p, v in zip(self.model.parameters(), vector):
			p.data.sub_(2*R, v)
		loss = self.model._loss(input, target)
		grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
		
		for p, v in zip(self.model.parameters(), vector):
			p.data.add_(R, v)
		return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

	def reset_model_parameters(self):
		model = self.Net().cuda()
		self.model.copy_params_from(model)
		self.sync_symbolic_model()

	def reset_arch_parameters(self):
		self.logger.info(self.model.get_type())
		#if optimizee.model.get_type() is 'AutoNetwork':
		params = self.model.gen_alphas()
		self.model.set_arch_paramters(params)
		self.symbolic_model.set_arch_paramters(params)

	def reset_optimizer_parameters(self):
		self.optimizer.set_beta(self.optimizer.gen_beta())
		self.reset_optimizer_state()

	def reset_optimizer_state(self):
		for k in self.optimizer.state.keys():
			self.optimizer.state[k] = {}
		self.optimizer.param_groups[0]['lr'] = args.learning_rate

	def state_dict(self):
		state_dict = {}
		state_dict['model'] = self.model.state_dict()
		if args.arch == 'auto' and args.arch_training:
			state_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
		state_dict['beta_optimizer'] = self.beta_optimizer.state_dict()
		state_dict['optimizer'] = self.optimizer.state_dict()
		return state_dict

	def load_state_dict(self, state_dict):
		self.model.load_state_dict(state_dict['model'])
		self.symbolic_model = copy.deepcopy(self.model)
		if args.arch == 'auto' and args.arch_training:
			self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])
		self.beta_optimizer.load_state_dict(state_dict['beta_optimizer'])
		self.optimizer.load_state_dict(state_dict['optimizer'])
		
