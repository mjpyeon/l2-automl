import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import numpy as np
import pdb
torch.manual_seed(97)
class MetaOptimizer(optim.Optimizer):
	def __init__(self, params, lr=2e-3, beta=None):
		defaults = dict(lr=lr)
		self.m_decay = 0.9
		self.vr_decay = 0.999
		num_ops = [16,16,4,4,5]
		self.beta = [Variable(1e-3*torch.randn(num_).cuda(), requires_grad=True) for num_ in num_ops]
		#self.beta[0].data = torch.Tensor([0.5,0.5]).cuda()
		#self.beta[1].data = torch.Tensor([1,-1e8,-1e8,-1e8]).cuda()
		if type(beta) != type(None):
			for b, b_given in zip(self.beta, beta):
				b.data.copy_(b_given.data)
		self.lr = Variable(torch.Tensor([1e-3]).cuda(), requires_grad=True)
		self.sf = nn.Softmax(-1)
		self.eps = 1e-8
		super(MetaOptimizer, self).__init__(params, defaults)

	def hard_operand(self, idx, ops):
		return ops[idx]
	def hard_unary(self, idx, input):
		unary_funcs = {
			0:lambda x:x,
			1:lambda x:-x,
			2:lambda x:torch.sqrt(x + self.eps),
			3:lambda x:torch.sign(x)
		}
		return unary_funcs[idx](input)
	def hard_graph(self, ops, code):
		op1 = self.hard_operand(code[0], ops) # [1, dim]
		u1 = self.hard_unary(code[1], op1) # [1, dim]
		return u1

	def ele_mul(self, b, a):
		return (b * a.transpose(0, len(a.size())-1)).transpose(len(a.size())-1,0)

	def operand(self, beta, ops):
		beta_ = self.sf(beta*100)
		return self.ele_mul(beta_, ops).sum(0)

	def unary(self, beta, input):
		input = input.unsqueeze(0)
		beta_ = self.sf(beta*100)
		unary_funcs = torch.cat((input, -input, torch.sqrt(torch.abs(input) + 1e-12), torch.sign(input)),0)
		return self.ele_mul(beta_, unary_funcs).sum(0)

	def binary(self, beta, left, right):
		beta_ = self.sf(beta*100)
		left = left.unsqueeze(0)
		right = right.unsqueeze(0)
		binary_funcs = torch.cat((left+right,left-right, left*right,left/(right + self.eps), left), 0)
		return self.ele_mul(beta_, binary_funcs).sum(0)

	def graph(self, ops):
		ops = torch.autograd.Variable(ops, requires_grad=False) # [num_ops, dim]
		op1 = self.operand(self.beta[0], ops) # [1, dim]
		op2 = self.operand(self.beta[1], ops) # [1, dim]
		u1 = self.unary(self.beta[2], op1) # [1, dim]
		u2 = self.unary(self.beta[3], op2) # [1, dim]
		b1 = self.binary(self.beta[4], u1, u2)
		return b1

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
					state['m'] = torch.zeros_like(p.data)
					state['m_py'] = torch.zeros_like(p.data)
					state['v'] = torch.zeros_like(p.data)
					state['r'] = torch.zeros_like(p.data)
					state['constant1'] = torch.ones_like(p.data)
					state['constant2'] = torch.ones_like(p.data) * 2
					state['eps'] = torch.ones_like(p.data) * self.eps

				state['step'] += 1
				# update operands
				grad2 = grad * grad
				grad3 = grad2 * grad
				state['m'].mul_(self.m_decay).add_(1 - self.m_decay, grad)
				state['m_py'].mul_(self.m_decay).add_(1, grad)
				state['v'].mul_(self.vr_decay).add_(1 - self.vr_decay, grad2)
				state['r'].mul_(self.vr_decay).add_(1 - self.vr_decay, grad3)

				ops = [
						grad,
						grad2,
						grad3,
						state['m'],
						state['m_py'],
						state['v'],
						state['r'],
						torch.sign(grad),
						torch.sign(state['m']),
						state['constant1'],
						state['constant2'],
						state['eps'],
						1e-4*p.data,
						1e-3*p.data,
						1e-2*p.data,
						1e-1*p.data
					]
				ops = [op.unsqueeze(0) for op in ops]
				update = self.graph(torch.cat(ops, 0))
				if DoUpdate:
					p.data.add_(-group['lr'], update.data)	
				params_updates += [-group['lr'] * update]
		return params_updates

if __name__ == '__main__':
	def rosenbrock(tensor):
		x, y = tensor
		return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
	params = Variable(torch.Tensor([2, 1.5]).cuda(), requires_grad=True)
	#'''
	optim = torch.optim.Adam([params], lr=1e-3)
	#optim = torch.optim.SGD([params], lr=0.0001, momentum=0.9)
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




