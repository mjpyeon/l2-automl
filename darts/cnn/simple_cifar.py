# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from meta_optimizer import MetaOptimizer
import meta_optimizer_gbg as meta_gbg
from vgg_model import VGG
from dpn import DPN92
from operator import mul
from collections import deque
import copy
import argparse
import math
from model import NetworkCIFAR as Network
from mobile_net import MobileNetV2
import genotypes
import pdb
torch.backends.cudnn.deterministic = True
torch.manual_seed(977)
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--arch_weight_decay', type=float, default=0.1, help='weight decay for optim arch encoding')
parser.add_argument('--arch_learning_rate', type=float, default=0.002, help='learning rate for arch encoding')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size for training')
parser.add_argument('--model_grad_norm', type=float, default=1.0, help='max grad norm for model')
parser.add_argument('--beta_grad_norm', type=float, default=1.0, help='max grad norm for beta')
parser.add_argument('--log_freq', type=int, default=50, help='logging frequency')
parser.add_argument('--bptt_step', type=int, default=1, help='steps for bptt')
#parser.add_argument('--update_step', type=int, default=1, help='steps for update')
parser.add_argument('--iteration', type=int, default=3, help='training iterations')
parser.add_argument('--epoch', type=int, default=3, help='num of epoch for each training iteration and testing')
parser.add_argument('--test_freq', type=int, default=10, help='frequency in epoch for running testing while training')
parser.add_argument('--save_path', type=str, default='', help='folder path to save trained model & meta optim')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--use_darts_arch', action='store_true', default=False, help='use darts architecture')
args = parser.parse_args()
print(args)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MetaModel:
    def __init__(self, model):
        self.model = model
    def detach(self):
        def _iter_child(parent_module):
            for i, module in enumerate(parent_module.children()):
                # if has sub modules
                if(len(module._modules) > 0):
                    _iter_child(module)
                elif len(module._parameters) > 0:
                    for key,param in module._parameters.items():
                        if(type(param) != type(None)):
                            module._parameters[key] = param.detach()
        _iter_child(self.model)            
        
    def get_flat_params(self):
        params = []
        def _iter_child(parent_module):
            for i, module in enumerate(parent_module.children()):
                # if has sub modules
                if(len(module._modules) > 0):
                    _iter_child(module)
                elif len(module._parameters) > 0:
                    for key,param in module._parameters.items():
                        if(type(param) != type(None)):
                            params.append(param.view(-1))
        _iter_child(self.model)           
        
        return torch.cat(params)

    def set_flat_params(self, flat_params):
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
        
    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)
    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)


class MetaHelper(nn.Module):
    def __init__(self, model, backup_model):
        super(MetaHelper, self).__init__()
        self.meta_model = model
        self.backup_model = backup_model
        self.backup_loss = None
        self.backup_step_count = 0 # how many steps the backup has not been reseted
        self.consecutive_backups = False # how many consecutive restoring
    def reset(self, model):
        self.meta_model.detach()
        self.meta_model.model.zero_grad()
        self.meta_model.copy_params_from(model)
    def update(self, model):
        self.meta_model.copy_params_to(model)
    def meta_update(self, model_with_grads, updates):
        # Meta update itself
        flat_params = self.meta_model.get_flat_params()
        flat_updates = torch.cat([up.view(-1) for up in updates])
        new_flat_params = flat_params + flat_updates
        new_flat_params = new_flat_params.view(-1)
        self.meta_model.set_flat_params(new_flat_params)
        return self.meta_model.model
    def check_backup_and_reset(self,model,loss):
        # if first backup or loss is better than best loss or have not refresh backup for too long, update backup to this 
        if(self.consecutive_backups):
            self.backup_loss *= 2
        if(type(self.backup_loss) == type(None) or loss < self.backup_loss or (self.backup_step_count > 100 and loss < 4*self.backup_loss)):
            self.reset(model)
            self.update(self.backup_model)
            self.backup_loss = loss
            self.backup_step_count = 0
            self.consecutive_backups = False
            print('resetting backup')
        # else if loss is much worse than backup loss and not keep restoring backups, restore
        elif(loss > 4*self.backup_loss and loss > 2):
            print('restoring backup')
            self.reset(self.backup_model)
            self.update(model)
            self.consecutive_backups = True
        # else the loss is still okay, just reset model
        else:
            self.reset(model)
            self.consecutive_backups = False
        # higher the backup loss bar, when keeping restoring
        self.backup_step_count += 1


#'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#'''
if args.use_darts_arch:
    CIFAR_CLASSES = 10
    genotype = eval("genotypes.%s" % args.arch)
    Net = lambda: Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    print('Network DARTS')
else:
    #Net = lambda: VGG('VGG11')
    #print('Network ','VGG11')
    Net = MobileNetV2
    print('Network ','MobileNetV2')
    pass
def eval(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if args.use_darts_arch and args.auxiliary:
                outputs, _ = model(images)
            else:
                outputs = model(images)
                if type(outputs) == type(()):
                    outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return float(correct) / total

#net_params = list(net.parameters())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss()
config = 'beta optimized, decouple update, exact bptt, post trainingng with learned beta'
print('optim adam w decay/clip')
print(config)
old_beta = None
#for i in range(1):
#    dummy = Net()
#''' Train with our differentiable graph meta optimizer
meta_helper = MetaHelper(MetaModel(Net().to(device)), Net().to(device))
for iteration in range(args.iteration):
    model = Net().to(device)
    if type(old_beta) == type(None):
        meta_optim = MetaOptimizer(model.parameters(), lr=args.learning_rate)
    else:
        meta_optim = MetaOptimizer(model.parameters(), lr=args.learning_rate, beta=old_beta)
    optimizer = optim.Adam(meta_optim.beta+[meta_optim.lr], lr=args.arch_learning_rate, weight_decay=args.arch_weight_decay)
    meta_helper.reset(model)
    #new_params = [p.clone() for p in net_params]
    #pdb.set_trace()
    beta_code = [np.argmax(beta.data.cpu().numpy(), axis=0) for beta in meta_optim.beta]
    print('Greedy beta code: {}'.format(beta_code))
    prev_new_loss = deque(maxlen=100)
    prev_new_loss.append(2.2)
    try:
        for epoch in range(args.epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            bptt_loss_sum = 0.0
            running_new_loss = 0.0
            loss_sum = 0.0
            #update_step_idx = 0
            model.drop_path_prob = args.drop_path_prob * epoch / float(args.epoch)
            meta_helper.meta_model.model.drop_path_prob = args.drop_path_prob * epoch / float(args.epoch)
            for i, data in enumerate(trainloader, 0):
                #if(i > 15):
                #    break
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                model.zero_grad()
                # forward + backward + optimize
                if args.use_darts_arch and args.auxiliary:
                    outputs, logits_aux = model(inputs)
                    loss = criterion(outputs, labels) + criterion(logits_aux, labels)
                else:
                    outputs = model(inputs)
                    if type(outputs) == type(()):
                        outputs = outputs[0]
                    loss = criterion(outputs, labels)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.model_grad_norm)
                if any(param.grad.min().item() > 1e30 for param in model.parameters()):
                    print('nan gradient encountered in model params, restoring beta')
                    meta_helper.check_backup_and_reset(model, 1e12)
                    meta_optim.restore_backup()
                    model.zero_grad()
                    #pdb.set_trace()

                params_updates = meta_optim.step()
                #params_updates = [p.clamp(-0.1,0.1) for p in params_updates]
                model.zero_grad()
                meta_model = meta_helper.meta_update(model, params_updates)
                if args.use_darts_arch and args.auxiliary:
                    outputs, logits_aux = meta_model(inputs)
                    new_loss = criterion(outputs, labels) + criterion(logits_aux, labels)
                else:
                    outputs = meta_model(inputs)
                    if type(outputs) == type(()):
                        outputs = outputs[0]
                    new_loss = criterion(outputs, labels)
                if(math.isnan(new_loss.item())):
                    print('nan new loss')
                    #pdb.set_trace()
                new_loss_delta = new_loss - sum(prev_new_loss)/len(prev_new_loss)
                loss_sum += new_loss_delta
                ## update if loss gets smaller
                meta_helper.update(model)

                # print statistics
                running_loss += loss.item()
                bptt_loss_sum += loss.item()
                if not math.isnan(new_loss.item()):
                    prev_new_loss.append(new_loss.item())
                running_new_loss += new_loss.item()
                if((i+1) % args.bptt_step == 0):
                    optimizer.zero_grad()
                    loss_sum.backward()
                    # normalize bptt gradient
                    for beta in meta_optim.beta:
                       beta.grad /= args.bptt_step
                    meta_optim.lr.grad /= args.bptt_step
                    if(any(torch.isnan(beta.grad).max() for beta in meta_optim.beta) or math.isnan(meta_optim.lr.grad.item())): 
                        print('beta/lr gra nan, skip this update')
                        meta_helper.check_backup_and_reset(model, 1e12)
                        meta_optim.restore_backup()
                        optimizer.zero_grad()

                    grad_norm = nn.utils.clip_grad_norm_(meta_optim.beta, args.beta_grad_norm)
                    meta_optim.set_backup(loss_sum.item())
                    optimizer.step()
                    # randomly change beta if loss gets worse
                    #if loss_sum.item() > prev_loss_sum:
                    #    for beta, num_ in zip(meta_optim.beta, meta_optim.num_ops):
                    #        beta.data = 8e-3*torch.randn(num_).cuda()
                    if (math.isnan(new_loss.item())):
                        bptt_loss_sum = np.inf
                    meta_helper.check_backup_and_reset(model, bptt_loss_sum)
                    #meta_helper.reset(model)
                    if(any(torch.isnan(beta).max() for beta in meta_optim.beta) or math.isnan(meta_optim.lr.item())): 
                        print('beta/lr nan, should not happen!')
                        pdb.set_trace()
                    #meta_helper.reset(model)
                    loss_sum, bptt_loss_sum = 0.0, 0.0
                if (i+1) % args.log_freq == 0:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f, new loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / args.log_freq, running_new_loss/args.log_freq))
                    running_loss = 0.0
                    running_new_loss = 0.0
                if (i) % args.log_freq == 0:
                    print('beta[0] abs mean: {}'.format(meta_optim.beta[0].abs().mean()))

            beta_code = [np.argmax(beta.data.cpu().numpy(), axis=0) for beta in meta_optim.beta]
            print('Greedy beta code: {}'.format(beta_code))
            if (epoch+1) % args.test_freq == 0:
                print('Run Testing')
                acc = eval(model)
                # if accuracy not above random, discard this example and start a new one
                if(acc <= 0.11):
                    break
                if args.save_path != '':
                    torch.save(model, args.save_path+'/model_iter'+str(iteration))
                    torch.save(meta_optim, args.save_path+'/meta_optim_iter'+str(iteration))
                    print('Saving to {}'.format(args.save_path))
    except KeyboardInterrupt:
        pass
    old_beta = meta_optim.beta
    print('Finished Training')
    print('beta[0] abs mean: {}'.format(meta_optim.beta[0].abs().mean()))
    eval(model)

#'''
''' Train with traditional optimizer
print('Traditional Optimizer: SGD')
model = Net().to(device)
direct_optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
for epoch in range(args.epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for param in model.parameters():
            param.grad = param.grad * torch.sqrt(torch.abs(param.grad))
        direct_optim.step()
        running_loss += loss.item()
        if (i+1) % args.log_freq == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / args.log_freq))
            running_loss = 0.0
    if (epoch+1) % args.test_freq == 0:
        print('Run Testing')
        eval(model)
#'''
#print('Finished Training')
#eval(model)
#eval(meta_model)
'''
# use trained beta to train new model
print("test trained beta")
second_model = Net().to(device)
meta_optim = MetaOptimizer(second_model.parameters(), beta=old_beta)
#meta_helper = MetaHelper(MetaModel(Net().to(device)))
meta_helper.reset(second_model)
#update_step_idx = 0

for epoch in range(args.epoch):
    running_loss = 0.0
    second_model.drop_path_prob = args.drop_path_prob * epoch / float(args.epoch)
    meta_helper.meta_model.model.drop_path_prob = args.drop_path_prob * epoch / float(args.epoch)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        second_model.zero_grad()

        # forward + backward + optimize
        if args.use_darts_arch and args.auxiliary:
            outputs, logits_aux = second_model(inputs)
            loss = criterion(outputs, labels) + criterion(logits_aux, labels)
        else:
            outputs = second_model(inputs)
            if type(outputs) == type(()):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(second_model.parameters(), args.model_grad_norm)

        params_updates = meta_optim.step()
        #params_updates = [torch.clamp(up, min=-0.1, max=0.1) for up in params_updates]
        second_model.zero_grad()
        meta_helper.meta_update(second_model, params_updates)
        #if(update_step_idx >= args.update_step):
        meta_helper.update(second_model)
        #update_step_idx = 0
        #update_step_idx += 1
        meta_helper.reset(second_model)
        running_loss += loss.item()
        if (i+1) % args.log_freq == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / args.log_freq))
                print('beta[0] abs mean: {}'.format(meta_optim.beta[0].abs().mean()))
                running_loss = 0.0
    if (epoch+1) % args.test_freq == 0:
        print('Run Testing')
        eval(second_model)
eval(second_model)
#'''
