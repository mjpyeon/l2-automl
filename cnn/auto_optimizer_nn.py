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
import itertools


class LSTMOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super(LSTMOptimizer, self).__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        
    def forward(self, inp, hidden, cell):
    	inp = torch.autograd.Variable(inp, requires_grad=False)
        if self.preproc:
            # Implement preproc described in Appendix A
            
            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = w(torch.zeros(inp.size()[0], 2))
            keep_grads = torch.abs(inp) >= self.preproc_threshold
            inp2[:, 0][keep_grads] = torch.log(torch.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor
            inp2[:, 1][keep_grads] = torch.sign(inp[keep_grads])
            
            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = float(np.exp(self.preproc_factor)) * inp[~keep_grads]
            inp = w(Variable(inp2))

        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

class AutoOptimizer(optim.Optimizer):
	def __init__(self, params, lr=2e-3):
		defaults = dict(lr=lr)
		self.logger = L.get_logger(args.logdir)
		self.optimizer = LSTMOptimizer(args.preproc, args.opt_hidden).cuda()
		super(AutoOptimizer, self).__init__(params, defaults)
	
	def detach(self):
		# detach the hidden states and cell states
		for group in self.param_groups:
			for p in group['params']:
				state = self.state[p]
				if len(state) != 0:
					for lstm_states in state.values():
						for lstm_state in lstm_states:
							lstm_state.detach_()
	
	def optim_parameters(self):
		return self.optimizer.parameters()

	def state_dict(self, no_state = False):
		if no_state:
			state_dict = {}	
		else:
			state_dict = super(AutoOptimizer, self).state_dict()
		state_dict['optimizer'] = self.optim_parameters()
		return state_dict

	def load_state_dict(self, state_dict, no_state = False):
		if('state' in state_dict and no_state is False):
			super(AutoOptimizer, self).load_state_dict(state_dict)
		for saved_p, p in zip(state_dict['optimizer'], self.optim_parameters()):
			p.data.copy_(saved_p.data)
	
	def step(self, do_update=False, virtual=False):
		loss = None
		params_updates = []
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				state = self.state[p]
				cur_sz = int(np.prod(p.size()))
				# State initialization
				if len(state) == 0:
					state['hidden_states'] = [Variable(torch.zeros(cur_sz, args.opt_hidden)).cuda() for _ in range(2)]
					state['cell_states'] = [Variable(torch.zeros(cur_sz, args.opt_hidden)).cuda() for _ in range(2)]
				update, new_hidden_states, new_cell_states = self.optimizer(grad.view(cur_sz, 1), state['hidden_states'], state['cell_states'])
				if not virtual:
					state['hidden_states'] = new_hidden_states
					state['cell_states'] = new_cell_states
				update = update.view(*p.size())
				if do_update:
					p.data.add_(-group['lr'], update.data)	
				params_updates += [-group['lr'] * update]
		return params_updates

if __name__ == '__main__':
	def convex_curve(x):
		return (x[0]-5) ** 2 + (x[1]+5) ** 2
	torch.manual_seed(12)
	params = Variable(torch.Tensor([0.1, 0.1]).cuda(), requires_grad=True)
	optimizer = AutoOptimizer([params], lr=1e0)
	meta_optim = torch.optim.SGD(optimizer.optimizer.parameters(), lr=1e-1)
	avg_loss = collections.deque(maxlen=50)
	for i in range(4000):
		optimizer.zero_grad()
		loss = convex_curve(params)
		avg_loss.append(loss.item())
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_([params], 10.)
		
		param_updates = optimizer.step()
		next_step_params = params + param_updates[0]
		next_step_loss = convex_curve(next_step_params)
		if True or next_step_loss < loss:
			params.data.copy_(next_step_params.data)
		
		meta_optim.zero_grad()
		next_step_loss.backward()
		'''
		if param_updates[0].abs().item() > 1e1: # too large update
			proxy = sum([choice.max() for choice in choices]) * 1e2 # use choices.sum() as proxy to bp
			proxy.backward()
		else:
			(sum([choice.max() for choice in choices]) * (next_step_loss.item() - sum(avg_loss)/len(avg_loss))).backward() # REINFORCE 
		'''
		grad_norm = nn.utils.clip_grad_norm_(optimizer.optimizer.parameters(), 10.)
		meta_optim.step()
		optimizer.detach()


		if((i+1) % 20 == 0):
			#with torch.no_grad():
			#	choice_list = [c.max(0)[1].item() for c in choices]
			print('i: {}, loss: {:.2e}, new loss: {:.2e}/{:.2e}'.format(i, loss.item(), next_step_loss.item(), grad_norm))

		for b in optimizer.optimizer.parameters():
			if math.isnan(b.grad.mean().item()):
				print('opt lstm grad nan')
				pdb.set_trace()

		#if (i >= 476):
		#	pdb.set_trace()
		#	pass
		
		# [(i, optimizer.beta[i].grad.mean().item()) for i in range(20)]
		# optimizer.beta_graph_history[::-1]

	# test with trained beta
	params.data.copy_(torch.Tensor([0.1]).cuda())
	for k in optimizer.state.keys():
		optimizer.state[k] = {}
	for i in range(2000):
		optimizer.zero_grad()
		loss = convex_curve(params)
		avg_loss.append(loss.item())
		loss.backward()
		grad_norm = nn.utils.clip_grad_norm_([params], 10.)
		param_updates = optimizer.step(do_update=True)
		
		if((i+1) % 100 == 0):
			print('i: {}, loss: {:.2e}'.format(i, loss.item()))