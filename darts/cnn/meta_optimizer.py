import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import numpy as np
import pdb

class MetaOptimizer(optim.Optimizer):
	def __init__(self, params, lr=1e-3):
		defaults = dict(lr=lr)
		self.m_decay = 0.9
		num_ops = [2,4]
		self.beta = [Variable(1e-3*torch.randn(num_ops[i]).cuda(), requires_grad=True) for i in range(2)]
		self.lr = Variable(torch.Tensor([1e-3]).cuda(), requires_grad=True)
		self.sf = nn.Softmax(-1)
		self.eps = 1e-8
		super(MetaOptimizer, self).__init__(params, defaults)

	def ele_mul(self, b, a):
		return (b * a.transpose(0, len(a.size())-1)).transpose(len(a.size())-1,0)

	def operand(self, beta, ops):
		beta_ = self.sf(beta)
		return self.ele_mul(beta_, ops).sum(0)

	def unary(self, beta, input):
		input = input.unsqueeze(0)
		beta_ = self.sf(beta)
		unary_funcs = torch.cat((input, -input, torch.sqrt(torch.abs(input) + self.eps), torch.sign(input)),0)
		return self.ele_mul(beta_, unary_funcs).sum(0)

	def graph(self, ops):
		ops = torch.autograd.Variable(ops, requires_grad=False) # [num_ops, dim]
		op1 = self.operand(self.beta[0], ops) # [1, dim]
		u1 = self.unary(self.beta[1], op1) # [1, dim]
		return u1

	def step(self, DoUpdate=False, closure=None):
		loss = None
		if closure is not None:
			loss = closure()

		params_updates = []
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				state = self.state[p]
				# State initialization
				if len(state) == 0:
					state['step'] = 0
					# Exponential moving average of gradient values
					state['exp_avg'] = torch.zeros_like(p.data)

				exp_avg = state['exp_avg']

				state['step'] += 1
				# Decay the first moment running average coefficient
				exp_avg.mul_(self.m_decay).add_(1 - self.m_decay, grad)
				update = self.graph(torch.cat((grad.unsqueeze(0), state['exp_avg'].unsqueeze(0)), 0))
				if DoUpdate:
					p.data.add_(-group['lr'], update.data)	
				params_updates += [-group['lr'] * update]
		return params_updates

if __name__ == '__main__':
	def rosenbrock(tensor):
		x, y = tensor
		return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
	params = Variable(torch.Tensor([2, 1.5]).cuda(), requires_grad=True)
	'''
	optim = torch.optim.Adam([params], lr=1e-3)
	for i in range(2000):
		optim.zero_grad()
		loss = rosenbrock(params)
		loss.backward()
		optim.step()
		if(i % 50 == 0):
			print('i: {}, loss: {}'.format(i, loss.data[0]))
	'''
	meta_optim = MetaOptimizer([params])
	optim = torch.optim.Adam(meta_optim.beta, lr=1e-3)
	new_params = params.clone()
	for i in range(2000):
		meta_optim.zero_grad()
		loss = rosenbrock(params)
		loss.backward()
		params_updates = meta_optim.step(DoUpdate=(i % 2 == 0))
		new_params += params_updates[0]
		new_loss = rosenbrock(new_params)
		new_loss.backward(retain_graph=True)
		optim.step()
		if(i % 10 == 0):
			new_params = params.clone()
			optim.zero_grad()
		if(i % 50 == 0):
			print('i: {}, loss: {}, new loss: {}'.format(i, loss.item(), new_loss.item()))
		if(i % 500 == 0):
			print(meta_optim.beta)
	#'''




