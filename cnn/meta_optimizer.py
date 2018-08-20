import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import numpy as np
import pdb
from Logger import logger
#torch.manual_seed(977)
class MetaOptimizer(optim.Optimizer):
	def __init__(self, params, lr=2e-3, state_dict=None):
		defaults = dict(lr=lr)
		self.m_decay = 0.9
		self.vr_decay = 0.999
		self.num_ops = [14,14,6,6,5] + [15,15,6,6,5] + [16,16,6,6,5] + [17,17,6,6,5]
		self.min_beta_scaling = 100
		self.beta_scaling = 100
		self.logged_message = set()
		logger.log_once('Beta scaling: {}, Meta opt graph: {}'.format(self.beta_scaling, self.num_ops))
		self.cons = Variable(torch.Tensor([8e-3]).cuda(), requires_grad=False)
		logger.log_once('fixing beta 0 dim, 8e-3')
		#self.beta = [Variable(8e-3*torch.randn(num_).cuda(), requires_grad=True) for num_ in self.num_ops]
		self.init_beta()
		#for b in self.beta:
		#	b[0].data.copy_(torch.Tensor([8e-3]).squeeze())
		self.loss_check = 1e8
		self.lr_scaling = Variable(torch.Tensor([8e-3]).cuda(), requires_grad=True)
		if type(state_dict) != type(None):
			self.load_state_dict(state_dict)
		logger.log_once('using self.lr_scaling')
		self.sf = nn.Softmax(-1)
		self.tanh = nn.Tanh()
		self.eps = 1e-8
		self.backup_loss = 1e12
		super(MetaOptimizer, self).__init__(params, defaults)

	def init_beta(self):
		self.beta = [Variable(8e-3*torch.randn(num_-1).cuda(), requires_grad=True) for num_ in self.num_ops]
		#logger.log_once('self.beta = [Variable(8e-3*torch.randn(num_).cuda(), requires_grad=True) for num_ in self.num_ops]')

	def training_params(self):
		return self.beta + [self.lr_scaling]

	def state_dict(self, NoStates=False):
		if NoStates:
			state_dict = {}	
		else:
			state_dict = super(MetaOptimizer, self).state_dict()
		state_dict['beta'] = self.beta
		state_dict['lr_scaling'] = self.lr_scaling
		return state_dict

	def load_state_dict(self, state_dict, NoStates=False):
		if('state' in state_dict and NoStates != True):
			super(MetaOptimizer, self).load_state_dict(state_dict)
		for beta, beta_save in zip(self.beta, state_dict['beta']):
			beta.data.copy_(beta_save.data)
		self.lr_scaling.data.copy_(state_dict['lr_scaling'].data)
		self.print_beta_greedy()

	def beta_decay(self, decay_rate):
		for beta in self.beta:
			beta.data.add_(-decay_rate, beta.data)

	def set_backup(self, loss):
		if loss < self.backup_loss:
			self.backup_loss = loss
			self.backup_beta_data = [self.beta[i].data.clone() for i in range(len(self.beta))]
			self.lr_scaling_backup = self.lr_scaling.data.clone()
	def restore_backup(self):
		for beta, backup_beta in zip(self.beta, self.backup_beta_data):
			beta.data.copy_(backup_beta)
		self.lr_scaling.data.copy_(self.lr_scaling_backup)
		self.backup_loss *= 2

	def encourage_reuse(self):
		for i, beta in enumerate(self.beta):
			if((i % 5 == 0 or i % 5 == 1) and i / 5 > 0):
				max_beta = beta.max().data
				beta[15:].data.add_(1e-3*(max_beta - beta[15:].data))

	def beta_entropy(self, loss_ratio):
		loss = 0.0
		for i in range(len(self.beta)):
			#beta = self.beta[i]
			beta = torch.cat([self.cons, self.beta[i]])
			beta_ = self.sf(beta*self.beta_scaling) + 1e-12
			loss += (beta_ * beta_.log()).sum()
			entropy_loss = (beta_ * beta_.log()).sum()
			beta_grad = torch.autograd.grad(entropy_loss, self.beta[i])[0]
			beta_grad_ratio = (self.beta[i].grad.abs().mean().item() / (beta_grad.abs().mean().item()) + 1e-12) * loss_ratio
			self.beta[i].grad += beta_grad_ratio * beta_grad

	def hard_operand(self, idx, ops):
		return ops[idx]
	def hard_unary(self, idx, input):
		unary_funcs = {
			0:lambda x:x,
			1:lambda x:-x,
			2:lambda x:torch.sqrt(torch.abs(x) + self.eps),
			3:lambda x:torch.sign(x)
		}
		return unary_funcs[idx](input)
	def hard_binary(self, idx, left, right):
		binary_funcs = {
			0:lambda x,y:x+y,
			1:lambda x,y:x-y,
			2:lambda x,y:x*y,
			3:lambda x,y:x/(y+self.eps)
		}
		return binary_funcs[idx](left, right)
	def hard_graph(self, ops, code):
		op1 = self.hard_operand(code[0], ops) # [1, dim]
		op2 = self.hard_operand(code[1], ops) # [1, dim]
		u1 = self.hard_unary(code[2], op1) # [1, dim]
		u2 = self.hard_unary(code[3], op2) # [1, dim]
		b1 = self.hard_binary(code[4], u1, u2) # [1, dim]
		return b1

	def ele_mul(self, b, a):
		return (b * a.transpose(0, len(a.size())-1)).transpose(len(a.size())-1,0)

	def transform_beta(self, beta, maskIdx=0, maskStart=0):
		beta = torch.cat([self.cons, beta])
		if(maskIdx > 0):
			beta[maskStart:maskIdx] = -np.inf
		beta_ = self.sf(beta*self.beta_scaling)
		return beta_

	def print_beta_greedy(self):
		with torch.no_grad():
			num_subgraph = len(self.num_ops) / 5
			beta_ = []
			for graph_idx in range(num_subgraph):
				start = graph_idx * 5
				ops_mask, binary_mask = 0, 0
				if(graph_idx > 0 and torch.max(self.beta[start+0],0)[1] < 13):
					ops_mask = 14
					binary_mask = 5
				op1_beta = self.transform_beta(self.beta[start+0]).data.cpu().numpy()
				op2_beta = self.transform_beta(self.beta[start+1], ops_mask).data.cpu().numpy()
				u1_beta = self.transform_beta(self.beta[start+2]).data.cpu().numpy()
				u2_beta = self.transform_beta(self.beta[start+3]).data.cpu().numpy()
				b1_beta = self.transform_beta(self.beta[start+4], binary_mask, 4).data.cpu().numpy()
				beta_ += [op1_beta, op2_beta, u1_beta, u2_beta, b1_beta]
			'''
			for i, beta in enumerate(self.beta):
				if(i % 5 == 1 and i > 1 and torch.max(self.beta[i-1],0)[1] < 13):
					maskIdx = 16
				else:
					maskIdx = 0
				#maskIdx = 0
				beta_ += [self.transform_beta(beta, maskIdx).data.cpu().numpy()]
			'''
		beta_code = [np.argmax(b, axis=0) for b in beta_]
		beta_prob = [b[idx] for b,idx in zip(beta_, beta_code)]
		beta_prob = [np.prod(beta_prob[5*graph_idx:5*(graph_idx+1)]) for graph_idx in range(num_subgraph)]
		logger.log('Greedy beta code: {}, subgraph prod: {}'.format(beta_code, beta_prob))

	def operand(self, beta, ops, maskIdx=0):
		beta_ = self.transform_beta(beta, maskIdx)
		return self.ele_mul(beta_, ops).sum(0)

	def unary(self, beta, input):
		input = input.unsqueeze(0)
		beta_ = self.transform_beta(beta)
		unary_funcs = torch.cat((input, -input, torch.sqrt(torch.abs(input) + 1e-12), torch.sign(input), #), 0)
								torch.log(torch.abs(input).clamp(min=0.1) + self.eps), torch.exp(input.clamp(max=2))),0)
		return self.ele_mul(beta_, unary_funcs).sum(0)

	def binary(self, beta, left, right, maskIdx=0):
		beta_ = self.transform_beta(beta, maskIdx, 4)
		left = left.unsqueeze(0)
		right = right.unsqueeze(0)
		binary_funcs = torch.cat((left+right,left-right, left*right,(left/(right + self.eps)).clamp(min=-2, max=2), left),0)#, 
									#torch.abs(left).pow(right.clamp(max=5))), 0)
		return self.ele_mul(beta_, binary_funcs).sum(0)

	def graph(self, ops, start=0):
		#pdb.set_trace()
		#ops = torch.autograd.Variable(ops, requires_grad=False) # [num_ops, dim]
		ops_mask = 0
		binary_mask = 0
		logger.log_once('if(start > 1 and torch.max(self.beta[start+0],0)[1] < 13): ops_mask = 14')
		if(start > 1 and torch.max(self.beta[start+0],0)[1] < 13):
			ops_mask = 14
			binary_mask = 5
		op1 = self.operand(self.beta[start+0], ops) # [1, dim]
		op2 = self.operand(self.beta[start+1], ops, ops_mask) # [1, dim]
		u1 = self.unary(self.beta[start+2], op1) # [1, dim]
		u2 = self.unary(self.beta[start+3], op2) # [1, dim]
		b1 = self.binary(self.beta[start+4], u1, u2, binary_mask)
		if(math.isnan(b1.abs().mean().item()) or b1.abs().max().item() > 1e4):
			pdb.set_trace()
		return b1

	def hierarchical_graph(self, ops):
		ops = torch.autograd.Variable(ops, requires_grad=False) # [num_ops, dim]
		g1 = self.graph(ops, 0)
		ops = torch.cat([ops,g1.unsqueeze(0)])
		g2 = self.graph(ops, 5)
		ops = torch.cat([ops,g2.unsqueeze(0)])
		g3 = self.graph(ops, 10)
		ops = torch.cat([ops,g3.unsqueeze(0)])
		g4 = self.graph(ops, 15)
		
		#if(math.isnan(g4.abs().mean().item())):
		#	pdb.set_trace()
		return g4

	def step(self, DoUpdate=False, closure=None, virtual=False):
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
				if not virtual:
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
						#torch.sign(grad),
						torch.sign(state['m']),
						state['constant1'],
						state['constant2'],
						state['eps'],
						1e-4*p.data,
						1e-3*p.data,
						1e-2*p.data,
						#1e-1*p.data
					]
				ops = [op.unsqueeze(0) for op in ops]
				update = self.hierarchical_graph(torch.cat(ops, 0))#.clamp(min=-2, max=2)
				#update = self.hard_graph(torch.cat(ops, 0), [7, 2, 0, 2, 2])
				if DoUpdate:
					p.data.add_(-torch.exp(self.lr_scaling).item() * group['lr'], update.data)	
				params_updates += [-torch.exp(self.lr_scaling) * group['lr'] * update]
		return params_updates

if __name__ == '__main__':
	def rosenbrock(tensor):
		x, y = tensor
		return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
	params = Variable(torch.Tensor([2, 1.5]).cuda(), requires_grad=True)
	'''
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
	meta_optim = MetaOptimizer([params], lr=1e-3)
	optim = torch.optim.Adam(meta_optim.beta, lr=1e-3)
	best_params = params.data.clone()
	new_params = params.clone()
	running_loss, best_running_loss = 0.0, 1e8
	loss_sum = 0.0
	prev_loss = 0.0
	bptt_step = 10
	for i in range(2000):
		meta_optim.zero_grad()
		loss = rosenbrock(params)
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_([params], 10.)
		params_updates = meta_optim.step(DoUpdate=(i % 1 == 0))
		#pdb.set_trace()
		new_params = new_params + params_updates[0]
		new_loss = rosenbrock(new_params)
		loss_sum += new_loss - prev_loss
		prev_loss = new_loss.item()
		running_loss += loss.item()
		#new_loss.backward(retain_graph=True)
		if((i+1) % 1 == 0):
			print('i: {}, loss: {}, new loss: {}'.format(i, loss.item(), new_loss.item()))

		if((i+1) % bptt_step == 0):
			loss_sum.backward()
			for beta in meta_optim.beta:
				beta.grad /= bptt_step
			grad_norm = nn.utils.clip_grad_norm_(meta_optim.beta, 1.)
			optim.step()
			optim.zero_grad()
			#pdb.set_trace()
			# if this bptt loss is much higher than last time, restore model
			if(running_loss > 2*best_running_loss):
				params.data.copy_(best_params)
				print('restoring back up')
				pdb.set_trace()
			# else this update looks fine, set backup to this params
			else:
				if(running_loss < best_running_loss):
					best_running_loss = running_loss
				best_params.copy_(params.data)
				print('setting backup params')
			new_params = params.clone()
			running_loss, loss_sum = 0.0, 0.0

		if((i+1) % 500 == 0):
			print(meta_optim.beta)
	params.data = torch.Tensor([2, 1.5]).cuda()
	for i in range(2000):
		loss = rosenbrock(params)
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_([params], 10.)
		params_updates = meta_optim.step(DoUpdate=(i % 1 == 0))
		if(i % 50 == 0):
			print('i: {}, loss: {}'.format(i, loss.item()))

	#'''




