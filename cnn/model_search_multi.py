import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
from relax import to_onehot


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    weights_ = weights.data.cpu().numpy().astype(int)
    if weights_[1] == 1:
        return self._ops[weights_[0]](x)
    else:
        return 0



class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    # print(s0.shape, s1.shape)
    for i in range(self._steps):

      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input, arch):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = arch[1]#F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = arch[0]#F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    self.k = sum(1 for i in range(self._steps) for n in range(2+i))
    self.num_ops = len(PRIMITIVES)

    # self.alphas = Variable(1e-3*torch.randn(self.k*2, self.num_ops).cuda(), requires_grad=True)
    # self._arch_parameters = [self.alphas]

    self.alphas = []
    for i in range(self.k*2):
        self.alphas.append(Variable(1e-3*torch.randn(self.num_ops-1).cuda(), requires_grad=True))

    self.betas = []
    start = 0
    nums = 2
    q_dim = self.k * 2 * (self.num_ops - 1)
    while start < self.k*2:
        if start % self.k != 0:
            self.betas.append(Variable(1e-3*torch.randn(nums).cuda(), requires_grad=True))
            q_dim += nums * (nums - 1) / 2
        start = start+nums
        nums += 1
        if start == self.k:
            nums = 2
    self.q_dim = q_dim
    #self.disc_dim = self.k * (self.num_ops) * 2 - 4
    # self.arch_params = Variable(1e-3*torch.randn(self.k*2, self.num_ops).cuda(), requires_grad=True) # check whether the init is right
    # self.betas = []
    # for i in range(1, self._steps):
    #     self.betas.append(Variable(1e-3*torch.randn(i+2).cuda(), requires_grad=True))
    #     self.disc_dim += (i+2) * (i + 1)
    # for i in range(1, self._steps):
    #     self.betas.append(Variable(1e-3*torch.randn(i+2).cuda(), requires_grad=True))

    self._arch_parameters = self.alphas+self.betas #torch.cat([para.view(-1) for para in [self.alphas] + self.betas])

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def genotype_binary(self):
    def _parse(weights, bbb):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        tmp_bbb = bbb[start:end].copy()
        for j in range(i + 2):
          if int(tmp_bbb[j]) == 1:
              gene.append((PRIMITIVES[int(W[j])], j))
        start = end
        n += 1
      return gene

    sample_betas = []
    for i, beta in enumerate(self.betas):
        tmp = beta.topk(2)[1]
        sample_beta = torch.zeros_like(beta)
        sample_beta[tmp[0]] = 1
        sample_beta[tmp[1]] = 1
        if i == 0 or i == len(self.betas)/2:
            sample_betas.append(Variable(torch.Tensor(2).cuda().fill_(1), requires_grad=False))
        sample_betas.append(sample_beta)
    hard_betas = torch.cat(sample_betas)
    hard_samples = torch.max(torch.stack(self.alphas),1)[1].float()

    gene_normal = _parse(hard_samples[:self.k].data.cpu().numpy()+1, hard_betas[:self.k].data.cpu().numpy())
    gene_reduce = _parse(hard_samples[self.k:].data.cpu().numpy()+1, hard_betas[self.k:].data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    arch = torch.cat([hard_samples.float().unsqueeze(1) + 1, hard_betas.unsqueeze(1)], 1)
    return genotype, arch.split(self.k)
