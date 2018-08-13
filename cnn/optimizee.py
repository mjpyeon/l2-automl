import copy
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

"""
In Optimizee class, we keep two copies of the optimizee model:
    `model`: the model that can be BPed by L and updated by meta_optimizer
    `symbolic_model`: the model that is used as a variable in the computational graph, and BPed by L'. Its parameters,
               however, can only be updated by copying from model (below the level of computational graph)
"""
class Optimizee:
	def __init__(self, model):
	    self.model = model 
	    self.symbolic_model = copy.deepcopy(model)
	
	def sync_symbolic_model(self):
		"""
		sync symbolic model with model
		"""
		self.symbolic_model = copy.deepcopy(self.model)
		'''
		self.symbolic_model.copy_params_from(self.model)
		if(sync_alpha):
			self.symbolic_model.set_arch_paramters(self.model.arch_parameters)
		'''

	def update(self, updates):
		"""
		non-differentiable update
		"""
		self.model.update(updates)
	
	def differentiable_update(self, updates):
		"""
		differentiable update	
		"""
		self.symbolic_model.differentiable_update(updates)

	def detach(self):
		"""
		This detaches the symbolic_model.parameters from the past path that generates it,
		in order to perform truncated BPTT within every T steps.
		i.e. this will be called every T steps 
		"""
		self.symbolic_model.detach()
		#for param in self.symbolic_model.parameters():
		#    param = param.detach()



#def test():
#    Net = MobileNetV2
#    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#    meta_helper = OptimizeeHelper(Optimizee(Net().to(device)), Net().to(device))
