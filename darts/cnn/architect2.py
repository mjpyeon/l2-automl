import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from model_search import Network
from operator import mul
import pdb

torch.manual_seed(977)
def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.alpha_learning_rate, betas=(0.5, 0.999), weight_decay=args.alpha_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    dtheta = _concat(network_optimizer.step(virtual=True)) + self.network_weight_decay*theta
    model_unrolled = self._construct_model_from_theta(theta.sub(eta, dtheta))
    network_optimizer.zero_grad()
    return model_unrolled

  def _update_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    loss.backward()
    theta = _concat(self.model.parameters()).data
    dtheta = _concat(network_optimizer.step(virtual=True)) + self.network_weight_decay*theta
    theta = theta.sub(eta, dtheta)
    self._update_model_from_theta(theta)


  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(
            input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)

    grad_norm = nn.utils.clip_grad_norm_(self.model.arch_parameters(), 10.)
    self.optimizer.step()
    return grad_norm

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    for v in self.model.arch_parameters():
      if v.grad is not None:
        v.grad.data.zero_()
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    #pdb.set_trace()
    original_theta = _concat(self.model.parameters()).data
    self._update_unrolled_model(input_train, target_train, eta, network_optimizer)
    loss = self.model._loss(input_valid, target_valid)
    #grads = torch.autograd.grad(loss, self.model.arch_parameters(), retain_graph=True)
    loss.backward()
    grads = [p.grad for p in self.model.arch_parameters()]
    theta = self.model.parameters()
    #dtheta = torch.autograd.grad(loss, model_unrolled.parameters())
    dtheta = network_optimizer.step(virtual=True)
    vector = [dt.add(self.network_weight_decay, t).data for dt, t in zip(dtheta, theta)]
    self._update_model_from_theta(original_theta)
    implicit_grads = self._hessian_vector_product(self.model, vector, input_train, target_train)
    #implicit_grads = self._hessian_vector_product(self.model, vector, input_train, target_train)
    for g, ig in zip(grads, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), grads):
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
  '''
  def _update_model_from_theta(self, theta):
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    self.model.load_state_dict(model_dict)
  '''
  def _construct_model_from_theta(self, theta):
    model_clone = Network(self.model._C, self.model._num_classes, self.model._layers, self.model._criterion).cuda()

    for x, y in zip(model_clone.arch_parameters(), self.model.arch_parameters()):
        x.data.copy_(y.data)
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_clone.load_state_dict(model_dict)
    return model_clone.cuda()

  def _hessian_vector_product(self, model, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)
    loss = model._loss(input, target)
    grads_p = torch.autograd.grad(loss, model.arch_parameters())
    del loss

    for p, v in zip(model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = model._loss(input, target)
    grads_n = torch.autograd.grad(loss, model.arch_parameters())
    del loss

    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

