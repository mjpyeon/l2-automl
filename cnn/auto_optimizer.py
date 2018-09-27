import math
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import logger as L
from args import args
import collections


class AutoOptimizer(optim.Optimizer):
	def __init__(self, params, lr=2e-3):
		defaults = dict(lr=lr)
		self.logger = L.get_logger(args.logdir)
		self.m_decay = 0.9
		self.vr_decay = 0.99
		self.num_ops = [16,16,6,6,5] + [17,17,6,6,5] + [18,18,6,6,5] + [19,19,6,6,5]
		self.beta = [Variable(torch.zeros(num_).cuda(), requires_grad=True) for num_ in self.num_ops]
		self.beta_inv_temp = args.min_beta_inv_temp
		self.gumbel_temp = args.max_gumbel_temp
		self.logger.info('Beta temperature: {}, auto-optimizer graph: {}'.format(self.beta_inv_temp, self.num_ops))
		#self.cons = Variable(torch.Tensor([8e-3]).cuda(), requires_grad=False)
		self.logger.info('fixing beta 0 dim, 8e-3')
		self.loss_check = 1e8
		#self.lr_scaling = Variable(torch.Tensor([8e-3]).cuda(), requires_grad=True)
		#self.logger.info('using self.lr_scaling')
		self.sf = nn.Softmax(-1)
		self.eps = 1e-8
		self.backup_loss = 1e12
		self.init_beta()
		self.beta_graph_history = []
		super(AutoOptimizer, self).__init__(params, defaults)
	
	def init_beta_gold(self):
		greedy_beta = [0,0,0,0,0] + [0,0,0,0,0] + [0,0,0,0,0] + [0,0,0,0,4] # SGD
		#greedy_beta = [0,0,0,0,0] + [0,0,0,0,0] + [9,5,0,2,3] + [18,0,0,0,2] # RMSprop
		self._init_beta(1e5, greedy_beta)
		self.print_beta_greedy()

	def init_beta(self):
		#greedy_beta = [0,0,0,0,0] + [0,0,0,0,0] + [9,5,0,2,3] + [18,0,0,0,2] # RMSprop
		greedy_beta = [0,0,0,0,0] + [0,0,0,0,0] + [0,0,0,0,0] + [0,18,0,0,4] # SGD
		self._init_beta(6*1e-2)#, greedy_beta)
		self.print_beta_greedy()


	def _init_beta(self, bias, greedy_beta=None):
		beta = [(8e-3*torch.randn(num_).cuda()) for num_ in self.num_ops]
		if greedy_beta is not None:
			self.logger.info('initializing beta with greedy beta and {} bias'.format(bias))
			for b_,greedy_b in zip(beta, greedy_beta):
				b_.data[greedy_b] += bias
		else:
			self.logger.info('initializing beta randomly')
		for b,b_ in zip(self.beta, beta):
			b.data = b_

	def set_beta(self, given_beta):
		for b, b_ in zip(self.beta, given_beta):
			b.data.copy_(b_.data)

	def optim_parameters(self):
		return self.beta

	def state_dict(self, no_state = False):
		if no_state:
			state_dict = {}	
		else:
			state_dict = super(AutoOptimizer, self).state_dict()
		state_dict['beta'] = self.beta
		#state_dict['lr_scaling'] = self.lr_scaling
		return state_dict

	def load_state_dict(self, state_dict, no_state = False):
		if('state' in state_dict and no_state is False):
			super(AutoOptimizer, self).load_state_dict(state_dict)
		for beta, beta_save in zip(self.beta, state_dict['beta']):
			beta.data.copy_(beta_save.data)
		#self.lr_scaling.data.copy_(state_dict['lr_scaling'].data)

	def beta_decay(self, decay_rate):
		for beta in self.beta:
			beta.data.add_(-decay_rate, beta.data)

	def beta_entropy(self, loss_ratio):
		loss = 0.0
		for i in range(len(self.beta)):
			beta = self.beta[i]
			#beta = torch.cat([self.cons, self.beta[i]])
			beta_ = self.sf(beta*self.beta_inv_temp) + 1e-12
			loss += (beta_ * beta_.log()).sum()
			entropy_loss = (beta_ * beta_.log()).sum()
			beta_grad = torch.autograd.grad(entropy_loss, self.beta[i])[0]
			beta_grad_ratio = self.beta[i].grad.abs().mean().item() / (beta_grad.abs().mean().item() + 1e-10) * loss_ratio
			self.beta[i].grad += beta_grad_ratio * beta_grad


	def ele_mul(self, b, a):
		return (b * a.transpose(0, len(a.size())-1)).transpose(len(a.size())-1,0)

	def print_choices(self, choices):
		with torch.no_grad():
			choice_list = [c.max(0)[1].item() for c in choices]
		self.logger.info('sampled choices: {}, {}, {}, {}'.format(choice_list[0:5], choice_list[5:10], choice_list[10:15], choice_list[15:20]))

	def greedy_sample_choices(self):
		choices = []
		for beta_idx in range(len(self.num_ops)):
			start_idx = 5 * (beta_idx // 5)
			ops_mask, binary_mask = self.check_beta_mask(start_idx)
			if((beta_idx + 1) % 5 == 0): # if binary operation
				choice = self.greedy_transform_beta(self.beta[beta_idx], binary_mask, 4)
			elif(beta_idx % 5 == 1): # second operand
				choice = self.greedy_transform_beta(self.beta[beta_idx], ops_mask)
			else:
				choice = self.greedy_transform_beta(self.beta[beta_idx])
			choices.append(choice)
		return choices

	def greedy_transform_beta(self,beta,maskIdx=0,maskStart=0):
		y = (beta * 1)
		if(maskIdx > 0):
			y[maskStart:maskIdx] = -np.inf
		shape = y.size()
		_, ind = y.max(dim=-1)
		y_hard = torch.zeros_like(y).view(-1, shape[-1])
		y_hard.scatter_(1, ind.view(-1, 1), 1)
		y_hard = y_hard.view(*shape).detach()
		return y_hard

	def gumble_sample_choices(self):
		choices = []
		for beta_idx in range(len(self.num_ops)):
			start_idx = 5 * (beta_idx // 5)
			ops_mask, binary_mask = self.check_beta_mask(start_idx)
			if((beta_idx + 1) % 5 == 0): # if binary operation
				choice = self.gumble_transform_beta(self.beta[beta_idx], binary_mask, 4)
			elif(beta_idx % 5 == 1): # second operand
				choice = self.gumble_transform_beta(self.beta[beta_idx], ops_mask)
			else:
				choice = self.gumble_transform_beta(self.beta[beta_idx])
			choices.append(choice)
		return choices

	def gumble_transform_beta(self, beta, maskIdx=0, maskStart=0):
		def sample_gumbel(shape, eps=1e-20):
			U = torch.rand(shape).cuda()
			return -Variable(torch.log(-torch.log(U + eps) + eps))

		def gumbel_softmax_sample(logits, temperature):
			sample_logits = sample_gumbel(logits.size()) # up to 4-5, 60% at temp 1
			y = logits + sample_logits
			return F.softmax(y / temperature, dim=-1)

		def gumbel_softmax(logits, temperature):
			"""
			ST-gumple-softmax
			input: [*, n_class]
			return: flatten --> [*, n_class] an one-hot vector
			"""
			y = gumbel_softmax_sample(logits, temperature)
			shape = y.size()
			_, ind = y.max(dim=-1)
			y_hard = torch.zeros_like(y).view(-1, shape[-1])
			y_hard.scatter_(1, ind.view(-1, 1), 1)
			y_hard = y_hard.view(*shape)
			y_hard = (y_hard - y).detach() + y
			if(math.isnan(y_hard.mean().item())):
				pdb.set_trace()
				pass
			return y_hard
		#beta = torch.cat([self.cons, beta])
		beta = (beta * 1)
		if(maskIdx > 0):
			beta[maskStart:maskIdx] = -np.inf
		return gumbel_softmax(beta * self.beta_inv_temp, self.gumbel_temp) # fixed gumbel temperature for now, TODO anneal gumbel temp

	def transform_beta(self, beta, maskIdx=0, maskStart=0):
		#beta = torch.cat([self.cons, beta])
		beta = (beta * 1)
		if(maskIdx > 0):
			beta[maskStart:maskIdx] = -np.inf
		beta_ = self.sf(beta*self.beta_inv_temp)
		return beta_

	def print_beta_greedy(self):
		with torch.no_grad():
			num_subgraph = len(self.num_ops) / 5
			beta_ = []
			for graph_idx in range(num_subgraph):
				start = graph_idx * 5
				ops_mask, binary_mask = self.check_beta_mask(start)
				op1_beta = self.transform_beta(self.beta[start+0]).data.cpu().numpy()
				op2_beta = self.transform_beta(self.beta[start+1], ops_mask).data.cpu().numpy()
				u1_beta = self.transform_beta(self.beta[start+2]).data.cpu().numpy()
				u2_beta = self.transform_beta(self.beta[start+3]).data.cpu().numpy()
				b1_beta = self.transform_beta(self.beta[start+4], binary_mask, 4).data.cpu().numpy()
				beta_ += [op1_beta, op2_beta, u1_beta, u2_beta, b1_beta]
			'''
			for i, beta in enumerate(self.beta):
				if(i % 5 == 1 and i > 1 and torch.max(self.beta[i-1],0)[1] < 15):
					maskIdx = 16
				else:
					maskIdx = 0
				#maskIdx = 0
				beta_ += [self.transform_beta(beta, maskIdx).data.cpu().numpy()]
			'''
		beta_code = [np.argmax(b, axis=0) for b in beta_]
		beta_prob = [b[idx] for b,idx in zip(beta_, beta_code)]
		beta_prob = [np.prod(beta_prob[5*graph_idx:5*(graph_idx+1)]) for graph_idx in range(num_subgraph)]
		#self.logger.info('Greedy beta code: {}, subgraph prod: {}'.format(beta_code, beta_prob))
		print('\tGreedy beta code: {}, subgraph prod: {}'.format(beta_code, beta_prob))

	def operand(self, beta, ops, maskIdx=0):
		beta_ = self.transform_beta(beta, maskIdx)
		return self.ele_mul(beta_, ops).sum(0)

	def unary(self, beta, input):
		input = input.unsqueeze(0)
		beta_ = self.transform_beta(beta)
		unary_funcs = torch.cat((input, -input, torch.sqrt(torch.abs(input) + 1e-12), torch.sign(input), #), 0)
								torch.log(torch.abs(input) + self.eps), torch.exp(input)),0)
		return self.ele_mul(beta_, unary_funcs).sum(0)

	def binary(self, beta, left, right, maskIdx=0):
		beta_ = self.transform_beta(beta, maskIdx, 4)
		left = left.unsqueeze(0)
		right = right.unsqueeze(0)
		binary_funcs = torch.cat((left+right,left-right, left*right,left/(right + self.eps), left),0)#, 
									#torch.abs(left).pow(right.clamp(max=5))), 0)
		return self.ele_mul(beta_, binary_funcs).sum(0)

	def check_beta_mask(self, graph_start_idx):
		ops_mask, binary_mask = 0, 0
		# not first graph and first op not reuse, set mask
		if(graph_start_idx == 15 and torch.max(self.beta[graph_start_idx+0],0)[1] < 16):
			ops_mask = 16
			#binary_mask = 5
		return ops_mask, binary_mask

	def graph(self, ops, start=0):
		ops_mask, binary_mask = self.check_beta_mask(start)
		op1 = self.operand(self.beta[start+0], ops) # [1, dim]
		op2 = self.operand(self.beta[start+1], ops, ops_mask) # [1, dim]
		u1 = self.unary(self.beta[start+2], op1) # [1, dim]
		u2 = self.unary(self.beta[start+3], op2) # [1, dim]
		b1 = self.binary(self.beta[start+4], u1, u2, binary_mask)
		#if(math.isnan(b1.abs().mean().item())):
		#	pdb.set_trace()
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
		return g4

	def gumble_hierarchical_graph(self, ops, choices):
		'''
		Args
			choices: shape as self.num_ops, intermediate tensor from gumble sampling
		'''
		def operand(choice, ops):
			#return self.ele_mul(choice, ops).sum(0)
			choice_one, choice_ind = choice.max(0)
			return ops[choice_ind] * choice_one

		def unary(choice, input):
			#input = input.unsqueeze(0)
			#unary_funcs = torch.cat((input, -input, torch.sqrt(torch.abs(input)), torch.sign(input), #), 0)
			#						torch.log(torch.abs(input) + self.eps), torch.exp(input)),0)
			unary_funcs = [
							lambda x:x,
							lambda x:-x,
							lambda x:torch.sqrt(torch.abs(x) + 1e-12),
							lambda x:torch.sign(x),
							lambda x:torch.log(torch.abs(x) + 1e-12),
							lambda x:torch.exp(x)
								]
			#return self.ele_mul(choice, unary_funcs).sum(0)
			choice_one, choice_ind = choice.max(0)
			return unary_funcs[choice_ind](input) * choice_one

		def binary(choice, left, right):
			#left = left.unsqueeze(0)
			#right = right.unsqueeze(0)
			#binary_funcs = torch.cat((left+right,left-right, left*right,left/(right + self.eps), left),0)#, 
			#							#torch.abs(left).pow(right.clamp(max=5))), 0)
			binary_funcs = [
							lambda x,y: x+y,
							lambda x,y: x-y,
							lambda x,y: x*y,
							lambda x,y: x/(y + 1e-12),
							lambda x,y: x
								]
			#return self.ele_mul(choice, binary_funcs).sum(0)
			choice_one, choice_ind = choice.max(0)
			return binary_funcs[choice_ind](left, right) * choice_one
		def graph(ops, choices):
			op1 = operand(choices[0], ops) # [1, dim]
			op2 = operand(choices[1], ops) # [1, dim]
			u1 = unary(choices[2], op1) # [1, dim] DEBUG WRONG
			u2 = unary(choices[3], op2) # [1, dim]
			b1 = binary(choices[4], u1, u2)
			'''
			def register_func(name):
				return lambda x : self.beta_graph_history.append({name: x.item()})
			#register_func = lambda x : self.beta_graph_history.append(x.item())
			for out in ['op1', 'op2', 'u1', 'u2', 'b1']:
				locals()[out].register_hook(register_func(out))
			'''
			return b1
		ops = torch.autograd.Variable(ops, requires_grad=False) # [num_ops, dim]
		g1 = graph(ops, choices[0:5])
		ops = torch.cat([ops,g1.unsqueeze(0)])
		g2 = graph(ops, choices[5:10])
		ops = torch.cat([ops,g2.unsqueeze(0)])
		g3 = graph(ops, choices[10:15])
		ops = torch.cat([ops,g3.unsqueeze(0)])
		g4 = graph(ops, choices[15:20])
		return g4

	def update_states(self, param_grads):
		for group in self.param_groups:
			for p,grad in zip(group['params'], param_grads):
				grad2 = grad * grad
				grad3 = grad2 * grad
				state = self.state[p]
				state['m'].mul_(self.m_decay).add_(1 - self.m_decay, grad)
				state['m_py'].mul_(self.m_decay).add_(1, grad)
				state['v'].mul_(self.vr_decay).add_(1 - self.vr_decay, grad2)
				state['r'].mul_(self.vr_decay).add_(1 - self.vr_decay, grad3)


	def step(self, do_update=False, closure=None, virtual=False, use_gumble=True, choices=None):
		loss = None
		if closure is not None:
			loss = closure()
		if choices is None:
			choices = self.gumble_sample_choices()
		params_updates = []
		params_grads = []
		self.beta_graph_history = []
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				params_grads.append(grad)
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

				if use_gumble:
					update = self.gumble_hierarchical_graph(torch.cat(ops, 0), choices)#.clamp(min=-2, max=2)
				else:
					update = self.hierarchical_graph(torch.cat(ops, 0))#.clamp(min=-2, max=2)
				#update = self.hard_graph(torch.cat(ops, 0), [7, 2, 0, 2, 2])
				if do_update:
					p.data.add_(-group['lr'], update.data)	
				params_updates += [-group['lr'] * update]
		return params_updates, choices, params_grads

if __name__ == '__main__':
	def convex_curve(x):
		return (x-1) ** 2
	torch.manual_seed(12)
	params = Variable(torch.Tensor([0.1]).cuda(), requires_grad=True)
	optimizer = AutoOptimizer([params], lr=1e-3)
	beta_optim = torch.optim.SGD(optimizer.beta, lr=1e-1)
	avg_loss = collections.deque(maxlen=50)
	for i in range(1000):
		choices = optimizer.gumble_sample_choices()
		optimizer.zero_grad()
		loss = convex_curve(params)
		avg_loss.append(loss.item())
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_([params], 10.)
		
		param_updates, choices, params_grad = optimizer.step(choices=choices)
		next_step_params = params + param_updates[0]
		next_step_loss = convex_curve(next_step_params)
		if next_step_loss < loss:
			params.data.copy_(next_step_params.data)
		
		beta_optim.zero_grad()
		#next_step_loss.backward()
		#'''
		if param_updates[0].abs().item() > 1e1: # too large update
			proxy = sum([choice.max() for choice in choices]) * 1e2 # use choices.sum() as proxy to bp
			proxy.backward()
		else:
			(sum([choice.max() for choice in choices]) * (next_step_loss.item() - sum(avg_loss)/len(avg_loss))).backward() # REINFORCE 
		#'''
		grad_norm = nn.utils.clip_grad_norm_(optimizer.beta, 10.)
		beta_optim.step()


		if((i+1) % 1 == 0):
			with torch.no_grad():
				choice_list = [c.max(0)[1].item() for c in choices]
			print('i: {}, loss: {:.2e}, new loss: {:.2e}/{:.2e}, choices: {}'.format(i, loss.item(), next_step_loss.item(), grad_norm, choice_list))
		if ((i+1) % 50 == 0):
			optimizer.print_beta_greedy()

		for b in optimizer.beta:
			if math.isnan(b.grad.mean().item()):
				print('beta grad nan')
				pdb.set_trace()

		#if (i >= 476):
		#	pdb.set_trace()
		#	pass
		
		# [(i, optimizer.beta[i].grad.mean().item()) for i in range(20)]
		# optimizer.beta_graph_history[::-1]

	# test with trained beta
	choices = optimizer.greedy_sample_choices()
	with torch.no_grad():
		choice_list = [c.max(0)[1].item() for c in choices]
	print('testing choices: {}'.format(choice_list))
	params.data.copy_(torch.Tensor([0.1]).cuda())
	for k in optimizer.state.keys():
		optimizer.state[k] = {}
	for i in range(500):
		optimizer.zero_grad()
		loss = convex_curve(params)
		avg_loss.append(loss.item())
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_([params], 10.)
		param_updates, choices, params_grad = optimizer.step(choices=choices, do_update = True)
		
		if((i+1) % 10 == 0):
			print('i: {}, loss: {:.2e}'.format(i, loss.item()))