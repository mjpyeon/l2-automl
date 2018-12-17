from __future__ import absolute_import
from __future__ import print_function

from itertools import product

import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import relu, binary_cross_entropy, softmax, log_softmax

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = softmax(x, dim=-1) * log_softmax(x, dim=-1)
        b = -1.0 * b.sum(-1).mean(0)
        return b

class HLoss1(nn.Module):
    def __init__(self):
        super(HLoss1, self).__init__()

    def forward(self, x):
        probs = torch.sigmoid(x)
        b = probs * torch.log(probs) + (1-probs) * torch.log1p(-probs)
        b = -1.0 * b.mean(0)
        return b

def to_onehot(aa, k):
    if len(aa.shape) >= 1:
        hard_samples_oh = Variable(torch.FloatTensor(aa.shape[0], k).cuda(), requires_grad=False)
        hard_samples_oh.zero_()
        hard_samples_oh.scatter_(1, aa.unsqueeze(1), 1)
    else:
        hard_samples_oh = Variable(torch.FloatTensor(k).cuda(), requires_grad=False)
        hard_samples_oh.zero_()
        hard_samples_oh.scatter_(0, aa, 1)
    return hard_samples_oh

class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, scale, hidden_size=50):
        super(QFunc, self).__init__()
        self.h1 = torch.nn.Linear(num_latents, hidden_size)
        self.nonlin = torch.nn.ReLU()
        self.h2 = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, 1)
        self.scale = scale

    def forward(self, z):
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        z = self.h1(z)
        z = self.nonlin(z)
        z = self.h2(z)
        z = self.nonlin(z)
        z = self.out(z) * self.scale
        return z


def loss_func(b, t):
    return ((b - t) ** 2).mean()


def _parse_args(args):
    parser = argparse.ArgumentParser(
        description='Toy experiment from backpropagation throught the void, '
        'written in pytorch')
    parser.add_argument(
        '--estimator', choices=['reinforce', 'relax', 'rebar'],
        default='reinforce')
    parser.add_argument('--rand-seed', type=int, default=42)
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=1)
    #parser.add_argument('--target', type=float, default=.499)
    parser.add_argument('--num-latents', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.01)
    return parser.parse_args(args)


def reinforce(f_b, b, logits, **kwargs):
    # log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    # d_log_prob = torch.autograd.grad(
    #     [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    # d_logits = f_b.unsqueeze(1) * d_log_prob
    # return d_logits

    log_prob = torch.distributions.Bernoulli(torch.sigmoid(logits)).log_prob(b)
    d_log_prob = torch.autograd.grad([log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0].detach()
    d_logits = f_b * d_log_prob
    return d_logits

def _get_z_tilde(logits, b, v):
    theta = torch.sigmoid(logits.detach())
    v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
    z_tilde = logits.detach() + torch.log(v_prime) - torch.log1p(-v_prime)
    assert(z_tilde.requires_grad==True)
    return z_tilde


def rebar(
        f_b, b, logits, z, v, eta, log_temp, target, loss_func=loss_func,
        **kwargs):
    z_tilde = _get_z_tilde(logits, b, v)
    temp = torch.exp(log_temp).unsqueeze(0)
    sig_z = torch.sigmoid(z / temp)
    sig_z_tilde = torch.sigmoid(z_tilde / temp)
    f_z = loss_func(sig_z, target)
    f_z_tilde = loss_func(sig_z_tilde, target)
    log_prob = torch.distributions.Bernoulli(logits=logits).log_prob(b)
    d_log_prob = torch.autograd.grad(
        [log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0]
    # d z / d logits = 1, so d f_z / d logits = d f_z / d z
    d_f_z = torch.autograd.grad(
        [f_z], [z], grad_outputs=torch.ones_like(f_z),
        create_graph=True, retain_graph=True)[0]
    d_f_z_tilde = torch.autograd.grad(
        [f_z_tilde], [z_tilde], grad_outputs=torch.ones_like(f_z_tilde),
        create_graph=True, retain_graph=True)[0]
    diff = f_b.unsqueeze(1) - eta * f_z_tilde.unsqueeze(1)
    d_logits = diff * d_log_prob + eta * (d_f_z - d_f_z_tilde)
    var_loss = (d_logits ** 2).mean()
    var_loss.backward()
    return d_logits.detach()


def relax(f_b, b, logits, z, v, log_temp, q_func, **kwargs):
    z_tilde = _get_z_tilde(logits, b, v)
    temp = torch.exp(log_temp)#.unsqueeze(0)
    sig_z = torch.sigmoid(z / temp)
    sig_z_tilde = torch.sigmoid(z_tilde / temp)
    # f_z = q_func(sig_z)#[:, 0]
    # f_z_tilde = q_func(sig_z_tilde)#[:, 0]

    f_z, f_z_tilde = q_func(torch.stack([sig_z, sig_z_tilde]))
    log_prob = torch.distributions.Bernoulli(torch.sigmoid(logits)).log_prob(b)
    #log_prob_sum = log_prob.sum()
    # log_prob1 = -torch.nn.functional.softplus(-logits) + (b-1) * logits
    # assert(np.all(np.isclose(log_prob1.data.numpy(), log_prob.data.numpy())))
    # #logits.requires_grad_(True)
    d_log_prob = torch.autograd.grad([log_prob], [logits], grad_outputs=torch.ones_like(log_prob))[0].detach()
    #d_log_prob_sum = torch.autograd.grad([log_prob_sum], [logits], grad_outputs=torch.ones_like(log_prob_sum))[0].detach()
    #assert(np.all(np.isclose(d_log_prob.data.numpy(), d_log_prob_sum.data.numpy())))
    # d_log_prob1 = (b - torch.sigmoid(logits)).detach()
    # assert(np.all(np.isclose(d_log_prob1.data.numpy(), d_log_prob.data.numpy())))
    # print(d_log_prob.requires_grad)
    #d_log_prob.requires_grad_(False)
    #print(d_log_prob)
    #d_log_prob.mean().backward()
    #print(logits.grad.numpy())
    #exit(1)
    # d z / d logits = 1, so d f_z / d logits = d f_z / d z
    d_f_z = torch.autograd.grad(
        [f_z], [z], grad_outputs=torch.ones_like(f_z),
        create_graph=True, retain_graph=True)[0]
    d_f_z_tilde = torch.autograd.grad(
        [f_z_tilde], [z_tilde], grad_outputs=torch.ones_like(f_z_tilde),
        create_graph=True, retain_graph=True)[0]
    diff = f_b - f_z_tilde
    d_logits = diff * d_log_prob + d_f_z - d_f_z_tilde
    var_loss = (d_logits ** 2).mean()
    # var_loss1 = ((diff * d_log_prob1 + d_f_z - d_f_z_tilde).mean(0) ** 2).mean()
    # assert(np.all(np.isclose(var_loss.detach().numpy(), var_loss1.detach().numpy())))
    var_loss.backward()
    # if logits.grad is not None:
    #     print(logits.grad.data.numpy())
    return d_logits.detach()


class RelaxOptimizer(object):
    def __init__(self, model, args):
      self.model = model
      self.disc_dim = model.disc_dim
      self.logit_optim = torch.optim.Adam([self.model.alphas],
          lr=args.arch_learning_rate) # , betas=(0.5, 0.999), weight_decay=args.arch_weight_decay; check if this is reasonable

      print("dim of discrete variables: ", self.disc_dim)
      self.log_temp = Variable(torch.ones(self.disc_dim).cuda()*np.log(0.5), requires_grad=True) # check whether the init is right
      self.q_func = QFunc(self.disc_dim, args.cv_hidden).cuda()
      self.tune_optim = torch.optim.Adam([self.log_temp] + list(self.q_func.parameters()), lr=args.cv_learning_rate)
      self.args = args

    def step(self, input_train, target_train, input_valid, target_valid, criterion, network_optimizer, unrolled):
        def _sparse_reg(tens):
            start = 0
            len = 2
            ret = 0
            while start + len <= tens.shape[0]:
                tmp = tens[start:start + len]
                ret += tmp.sum() - tmp.view(-1).topk(2)[0].sum()*2
                start = start + len
                len += 1
            #ent1 = binary_cross_entropy(tens, tens.detach())
            #ent = (-tens*torch.log(tens) - (1-tens) * torch.log1p(-tens)).mean()
            #assert(np.isclose(ent1.data.cpu().numpy(), ent.data.cpu().numpy()))
            #print(ret.data.cpu().numpy(), ent.data.cpu().numpy())
            return  ret
        self.logit_optim.zero_grad()
        self.tune_optim.zero_grad()
        network_optimizer.zero_grad()
        u = Variable(torch.rand(self.disc_dim).cuda(), requires_grad=False)
        v = Variable(torch.rand(self.disc_dim).cuda(), requires_grad=False)
        z = self.model.alphas.detach() + torch.log(u) - torch.log1p(-u)
        # we detach/reattach the gradient here b/c we don't want to
        # propagate through 'logits'
        z.requires_grad = True
        assert(z.requires_grad == True)
        b = z.gt(0.).type_as(z)
        arch = b.view([2*self.model.k, self.model.num_ops]).split(self.model.k)
        clf_logits = self.model(input_train, arch)
        clf_loss = criterion(clf_logits, target_train)

        # for ii in self.model.parameters():
        #     assert(ii.grad is None or ii.grad.cpu().data.numpy().sum() == 0)
        # assert(ii.grad is None or self.log_temp.grad.cpu().data.numpy().sum() == 0)
        # for ii in self.q_func.parameters():
        #     assert(ii.grad is None or ii.grad.cpu().data.numpy().sum() == 0)
        # assert(ii.grad is None or self.model.alphas.grad.cpu().data.numpy().sum() == 0)

        clf_loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)

        # assert(self.log_temp.grad is None or self.log_temp.grad.cpu().data.numpy().sum() == 0)
        # for ii in self.q_func.parameters():
        #     assert(ii.grad is None or ii.grad.cpu().data.numpy().sum() == 0)
        # assert(self.model.alphas.grad is None or self.model.alphas.grad.cpu().data.numpy().sum() == 0)
        reg = self.args.cv_reg_weight * b.sum()
        #print(clf_loss.detach(), reg.detach())
        d_logits = relax(f_b=clf_loss.detach() + reg.detach(), b=b, logits=self.model.alphas, z=z, v=v,log_temp=self.log_temp,q_func=self.q_func)
        # d_logits = reinforce(f_b=clf_loss.detach(), b=b, logits=self.model.alphas)
        # assert(np.any(self.log_temp.grad.cpu().data.numpy() != 0))
        # for ii in self.q_func.parameters():
        #     assert(np.any(ii.grad.cpu().data.numpy().sum() != 0))
        # assert(self.model.alphas.grad is None or self.model.alphas.grad.cpu().data.numpy().sum() == 0)

        #alphas_ = torch.sigmoid(self.model.alphas).view([2*self.model.k, self.model.num_ops]).split(self.model.k)
        #reg = (_sparse_reg(alphas_[0]) + _sparse_reg(alphas_[1])) * self.args.cv_reg_weight
        #reg.backward()
        #tmp = self.model.alphas.grad.data.cpu().numpy().copy()
        self.model.alphas.backward(d_logits)  # mean of batch
        #tmp1 = self.model.alphas.grad.data.cpu().numpy().copy()
        #assert(np.any(tmp != tmp1) and np.any(tmp != 0))
        # assert(np.any(self.model.alphas.grad.cpu().data.numpy().sum() != 0))

        network_optimizer.step()
        self.logit_optim.step()
        self.tune_optim.step()
        return clf_logits, clf_loss, reg


class ArmOptimizer(object):
    def __init__(self, model, args):
      self.model = model
      self.disc_dim = model.disc_dim
      self.logit_optim = torch.optim.Adam([self.model.alphas],
          lr=args.arch_learning_rate) # , betas=(0.5, 0.999), weight_decay=args.arch_weight_decay; check if this is reasonable

      print("dim of discrete variables: ", self.disc_dim)
      self.args = args

    def step(self, input_train, target_train, input_valid, target_valid, criterion, network_optimizer, unrolled):
        def _sparse_reg(tens):
            start = 0
            len = 2
            ret = 0
            while start + len <= tens.shape[0]:
                tmp = tens[start:start + len]
                ret += tmp.sum() - tmp.view(-1).topk(2)[0].sum()*2
                start = start + len
                len += 1
            #ent1 = binary_cross_entropy(tens, tens.detach())
            #ent = (-tens*torch.log(tens) - (1-tens) * torch.log1p(-tens)).mean()
            #assert(np.isclose(ent1.data.cpu().numpy(), ent.data.cpu().numpy()))
            #print(ret.data.cpu().numpy(), ent.data.cpu().numpy())
            return  ret

        def _loss(bb):
            clf_logits = self.model(input_train, bb.view([2*self.model.k, self.model.num_ops]).split(self.model.k))
            clf_loss = criterion(clf_logits, target_train) + bb.mean()
            return clf_logits, clf_loss

        self.logit_optim.zero_grad()
        network_optimizer.zero_grad()

        u = Variable(torch.rand(self.disc_dim).cuda(), requires_grad=False)
        P2 = torch.sigmoid(self.model.alphas)
        b1 = (u>(1-P2)).float()
        b2 = (u<P2).float()
        l1 = _loss(b1)[1].detach()
        l2 = _loss(b2)[1].detach()
        alphas_grads = (l1 - l2) * (u - 0.5)

        q_b = torch.distributions.Bernoulli(P2)
        b = q_b.sample()
        clf_logits, clf_loss = _loss(b)

        reg = (-torch.nn.functional.softplus(-self.model.alphas) + self.model.alphas * (torch.sigmoid(self.model.alphas) - 1)).mean() * self.args.arch_reg_weight#Variable(torch.Tensor([0]).cuda(), requires_grad=False)
        reg.backward()
        self.model.alphas.backward(alphas_grads)
        clf_loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)

        network_optimizer.step()
        self.logit_optim.step()
        return clf_logits, clf_loss, reg

class CategoricalRelaxOptimizer(object):
    def __init__(self, model, args):
      self.model = model
      self.logit_optim = torch.optim.Adam([self.model.alphas],
          lr=args.arch_learning_rate) # , betas=(0.5, 0.999), weight_decay=args.arch_weight_decay; check if this is reasonable

      self.log_temp = Variable(torch.zeros(self.model.k * 2).cuda(), requires_grad=True)
      self.scale = Variable(torch.ones(1).cuda(), requires_grad=True)
      self.q_func = QFunc(self.model.disc_dim, scale=self.scale, hidden_size=args.cv_hidden).cuda()
      self.cv_optim = torch.optim.Adam([self.log_temp, self.scale] + list(self.q_func.parameters()), lr=args.cv_learning_rate)
      self.args = args

    def step(self, input_train, target_train, input_valid, target_valid, criterion, network_optimizer, unrolled):
        def _make_samples():
            eps=1e-8
            u = Variable(torch.rand(self.model.k*2, self.model.num_ops).cuda(), requires_grad=False)
            u2 = Variable(torch.rand(self.model.k*2, self.model.num_ops).cuda(), requires_grad=False)
            temp = torch.exp(self.log_temp)
            logprobs = log_softmax(self.model.alphas, 1)
            g = -torch.log(-torch.log(u + eps) + eps)
            logprobs_z = logprobs + g
            hard_samples = torch.max(logprobs_z,1)[1]

            hard_samples_oh = Variable(torch.FloatTensor(hard_samples.shape[0], self.model.num_ops).cuda(), requires_grad=False)
            hard_samples_oh.zero_()
            hard_samples_oh.scatter_(1, hard_samples.unsqueeze(1), 1)

            g2 = -torch.log(-torch.log(u2 + eps) + eps)
            scores2 = logprobs + g2

            B = (scores2 * hard_samples_oh).sum(1, keepdim=True) - logprobs
            y = -1. * torch.log(u2) + torch.exp(-1. * B)
            g3 = -1. * torch.log(y)
            scores3 = g3 + logprobs
            logprobs_zt = hard_samples_oh * scores2 + ((-1. * hard_samples_oh) + 1.) * scores3
            return hard_samples_oh, softmax(logprobs_z / temp.unsqueeze(1), 1), softmax(logprobs_zt / temp.unsqueeze(1), 1)

        self.logit_optim.zero_grad()
        self.cv_optim.zero_grad()

        arch, z, z_tilde = _make_samples()

        network_optimizer.zero_grad()
        clf_logits = self.model(input_train, arch.split(self.model.k))
        clf_loss = criterion(clf_logits, target_train)

        clf_loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)
        network_optimizer.step()
        # clf_logits = Variable(torch.Tensor(input_train.shape[0], 10).cuda(), requires_grad=False)
        # clf_loss = -arch[:, -1].sum()


        neg_ent = -HLoss()(self.model.alphas) * self.args.arch_reg_weight
        entropy_grads = torch.autograd.grad(
            [neg_ent], [self.model.alphas], grad_outputs=torch.ones_like(neg_ent),
            create_graph=True, retain_graph=True)[0]

        f_z, f_z_tilde = self.q_func(torch.stack([z.view(-1), z_tilde.view(-1)]))
        ddiff_grads = torch.autograd.grad(
            [f_z-f_z_tilde], [self.model.alphas], grad_outputs=torch.ones_like(f_z),
            create_graph=True, retain_graph=True)[0]
        alphas_grads = ddiff_grads + (clf_loss.detach() - f_z_tilde) * (arch - softmax(self.model.alphas.detach(), 1))

        self.model.alphas.backward(alphas_grads.detach() + entropy_grads)
        self.logit_optim.step()

        var_loss = (alphas_grads ** 2).sum()
        var_loss.backward()
        self.cv_optim.step()
        return clf_logits, clf_loss, neg_ent


class ReinforceOptimizer(object):
    def __init__(self, model, args):
      self.model = model
      self.logit_optim = torch.optim.Adam(model.arch_parameters(),
          lr=args.arch_learning_rate) # , betas=(0.5, 0.999), weight_decay=args.arch_weight_decay; check if this is reasonable

      self.cv = None #Variable(torch.ones(1).cuda(), requires_grad=False)
      self.args = args
      self.counter = 0
      #self.rampdown_length = args.rampdown_length * 200
      self.history_logp = []
      self.history_f = []

    def step(self, input_train, target_train, input_valid, target_valid, criterion, network_optimizer, unrolled,ent_weight):

        def _sample_betas(betas, cal_ent=False):
            start = 0
            len = 2
            ret = []
            logp_betas = 0
            neg_ent_betas = 0
            nums = 0.
            while start + len <= betas.shape[0]:
                tmp = betas[start:start + len]
                if cal_ent:
                    neg_ent_betas += -HLoss()(tmp)
                probs1 = softmax(tmp)
                np_samples = torch.multinomial(probs1, 2, replacement=False)
                sample = torch.zeros_like(probs1)
                sample[np_samples[0]] = 1
                sample[np_samples[1]] = 1
                # print(sample)
                assert(np_samples[0].data.cpu().numpy() != np_samples[1].data.cpu().numpy())
                ret.append(sample.squeeze())
                logp_betas += torch.log(probs1[np_samples[0]] * probs1[np_samples[1]] * (1 / (1 - probs1[np_samples[0]]) + 1 / (1 - probs1[np_samples[1]])) + 1e-8)
                start = start + len
                len += 1
                if start == 14:
                    len = 2
                nums += 1.
            #print([ii.data.cpu().numpy() for ii in ret])
            assert(nums == 8)
            return torch.cat(ret), logp_betas, neg_ent_betas/nums
        self.logit_optim.zero_grad()
        network_optimizer.zero_grad()

        alphas = self.model.alphas
        alphas_sampler = torch.distributions.Categorical(softmax(alphas))
        betas = self.model.betas
        self.history_f = []
        self.history_logp = []
        for i in range(self.args.num_arch_samples):

            alphas_instance = alphas_sampler.sample()
            logp_alphas = alphas_sampler.log_prob(alphas_instance)
            betas_instance, logp_betas, neg_ent_betas = _sample_betas(betas, True if i == self.args.num_arch_samples - 1 else False)
            #print(logp_alphas, logp_betas)
            logp = logp_alphas.sum() + logp_betas
            self.history_logp.append(logp)

            arch = torch.cat([alphas_instance.float().unsqueeze(1) + 1, betas_instance.unsqueeze(1)], 1)
            clf_logits = self.model(input_train, arch.split(self.model.k))
            clf_loss = criterion(clf_logits, target_train)
            self.history_f.append(clf_loss)

        batch_f = torch.cat(self.history_f)
        batch_logp = torch.cat(self.history_logp)
        mean_f = batch_f.mean()
        var_f = batch_f.std()
        batch_f.mean().backward()
        nn.utils.clip_grad_norm(self.model.parameters(), self.args.grad_clip)
        network_optimizer.step()

        if self.cv is None:
            self.cv = [mean_f.detach(), var_f.detach()]
        else:
            self.cv[0] = self.cv[0] * self.args.smoothing_factor + mean_f.detach()*(1 - self.args.smoothing_factor)
            self.cv[1] = self.cv[1] * self.args.smoothing_factor + var_f.detach()*(1 - self.args.smoothing_factor)
        f = (batch_f.detach() - self.cv[0]) / (self.cv[1]+1e-8)
        #print(batch_f.data.cpu().numpy(), self.cv[0].data.cpu().numpy(), self.cv[1].data.cpu().numpy(), f.data.cpu().numpy())

        neg_ent_alphas = -HLoss()(self.model.alphas)
        neg_ent = (neg_ent_betas + neg_ent_alphas) * ent_weight
        # tmp1, tmp2 = torch.autograd.grad([logp], self.model.arch_parameters(), grad_outputs=torch.ones_like(logp))
        # hard_samples_oh = Variable(torch.FloatTensor(alphas_instance.shape[0], self.model.num_ops).cuda(), requires_grad=False)
        # hard_samples_oh.zero_()
        # hard_samples_oh.scatter_(1, alphas_instance.unsqueeze(1), 1)
        # tmp3,tmp4 = hard_samples_oh - softmax(self.model.alphas), betas_instance - torch.sigmoid(self.model.betas)
        # assert(np.all(np.isclose(tmp1.data.cpu().numpy(), tmp3.data.cpu().numpy())), [tmp1.data.cpu().numpy(), tmp3.data.cpu().numpy()])
        # assert(np.all(np.isclose(tmp2.data.cpu().numpy(), tmp4.data.cpu().numpy())))
        loss = (batch_logp*f).mean()+neg_ent
        alphas_grad, betas_grad = torch.autograd.grad([loss], self.model.arch_parameters(), grad_outputs=torch.ones_like(loss))
        self.model.alphas.backward(alphas_grad)
        self.model.betas.backward(betas_grad)
        self.logit_optim.step()
        self.counter += 1

        return clf_logits, mean_f, neg_ent


def run_toy_example(args=None):
    args = _parse_args(args)
    #print('Target is {}'.format(args.target))
    tt = [0.49, 0.499, 0.501, 0.51]
    args.num_latents = len(tt)
    target = Variable(torch.from_numpy(np.array(tt, dtype=np.float32)), requires_grad=False)
    #target.fill_(args.target)
    logits = Variable(torch.zeros(args.num_latents), requires_grad=True)
    eta = Variable(torch.ones(args.num_latents), requires_grad=True)
    log_temp = Variable(torch.from_numpy(
        np.array([.5] * args.num_latents, dtype=np.float32)), requires_grad=True)
    q_func = QFunc(args.num_latents)
    torch.manual_seed(args.rand_seed)
    if args.estimator == 'reinforce':
        estimator = reinforce
        tunable = []
    elif args.estimator == 'rebar':
        estimator = rebar
        tunable = [eta, log_temp]
    else:
        estimator = relax
        tunable = [log_temp] + list(q_func.parameters())
    logit_optim = torch.optim.Adam([logits], lr=args.lr)
    if tunable:
        tune_optim = torch.optim.Adam(tunable, lr=args.lr)
    else:
        tune_optim = None
    for i in range(args.iters):
        logit_optim.zero_grad()
        if tune_optim:
            tune_optim.zero_grad()
        u = Variable(torch.rand(args.num_latents), requires_grad=False)
        v = Variable(torch.rand(args.num_latents), requires_grad=False)
        z = logits.detach() + torch.log(u) - torch.log1p(-u)
        # we detach/reattach the gradient here b/c we don't want to
        # propagate through 'logits'
        z.requires_grad = True
        assert(z.requires_grad == True)
        b = z.gt(0.).type_as(z)
        f_b = loss_func(b, target)
        d_logits = estimator(
            f_b=f_b, b=b, u=u, v=v, z=z, target=target, logits=logits,
            log_temp=log_temp, eta=eta, q_func=q_func,
        )
        logits.backward(d_logits)  # mean of batch
        d_logits = d_logits.data.numpy()
        logit_optim.step()
        if tune_optim:
            tune_optim.step()
        thetas = torch.sigmoid(logits.detach()).data.numpy()
        loss = thetas * (1 - np.array(tt)) ** 2
        loss += (1 - thetas) * np.array(tt) ** 2
        loss = loss.mean()
        mean = d_logits.mean()
        std = d_logits.std()
        print(i, loss, thetas, torch.exp(log_temp).data.numpy())
        # print(
        #     'Iter: {} Loss: {:.03f} Thetas: {} Mean: {:.03f} Std: {:.03f} '
        #     'Temp: {:.03f}'.format(
        #         i, loss, thetas, mean, std, torch.exp(log_temp).data.numpy()[0])
        # )


if __name__ == '__main__':
    run_toy_example()
