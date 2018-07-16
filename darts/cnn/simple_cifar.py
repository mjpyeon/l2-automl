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
from operator import mul
import pdb

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MetaModel:
    def __init__(self, model):
        self.model = model
    def detach(self):
        for module in self.model.children():
            if(len(module._parameters) > 0):
                module._parameters['weight'] = module._parameters['weight'].detach()
                module._parameters['bias'] = module._parameters['bias'].detach()
    def get_flat_params(self):
        params = []

        for module in self.model.children():
            if len(module._parameters) > 0:
                params.append(module._parameters['weight'].view(-1))
                params.append(module._parameters['bias'].view(-1))
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0
        for i, module in enumerate(self.model.children()):
            if(len(module._parameters) > 0):
                weight_shape = module._parameters['weight'].size()
                bias_shape = module._parameters['bias'].size()

                weight_flat_size = reduce(mul, weight_shape, 1)
                bias_flat_size = reduce(mul, bias_shape, 1)

                module._parameters['weight'] = flat_params[
                    offset:offset + weight_flat_size].view(*weight_shape)
                module._parameters['bias'] = flat_params[
                    offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)

                offset += weight_flat_size + bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)
    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)


class MetaHelper(nn.Module):
    def __init__(self, model):
        self.meta_model = model
    def reset(self, model):
        self.meta_model.detach()
        self.meta_model.model.zero_grad()
        self.meta_model.copy_params_from(model)
    def meta_update(self, model_with_grads, updates, DoUpdate=False):
        # Meta update itself
        flat_params = self.meta_model.get_flat_params()
        flat_updates = torch.cat([up.view(-1) for up in updates])
        new_flat_params = flat_params + flat_updates
        new_flat_params = new_flat_params.view(-1)
        self.meta_model.set_flat_params(new_flat_params)
        if DoUpdate:
            self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

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

def eval(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


#net_params = list(net.parameters())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss()
bptt_step = 1
update_step = 1
print('bptt_step: {}, update_step: {}'.format(bptt_step, update_step))

#''' Train with our differentiable graph meta optimizer
meta_helper = MetaHelper(MetaModel(Net().to(device)))
model = Net().to(device)
meta_optim = MetaOptimizer(model.parameters())
optimizer = optim.SGD(meta_optim.beta, lr=0.01)
meta_helper.reset(model)
#new_params = [p.clone() for p in net_params]
#pdb.set_trace()
prev_loss = torch.zeros(1).cuda()
#'''
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    running_new_loss = 0.0
    loss_sum = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        model.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        params_updates = meta_optim.step()
        model.zero_grad()
        meta_model = meta_helper.meta_update(model, params_updates, DoUpdate=((i+1) % update_step == 0))
        
        outputs = meta_model(inputs)
        new_loss = criterion(outputs, labels)
        loss_sum += new_loss
        #optimizer.zero_grad()
        #new_loss.backward(retain_graph=True)
        #optimizer.step()
        prev_loss = new_loss.data
        if((i+1) % bptt_step == 0):
            optimizer.zero_grad()
            loss_sum.backward()
            # normalize bptt gradient
            for beta in meta_optim.beta:
                beta.grad /= bptt_step
            optimizer.step()
            meta_helper.reset(model)
            loss_sum = 0.0
            #prev_loss = torch.zeros(1).cuda()

        # print statistics
        running_loss += loss.item()
        running_new_loss += new_loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f, new loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000, running_new_loss/2000))
            running_loss = 0.0
            running_new_loss = 0.0
        #if i % 8000 == 0:
        #    print(meta_optim.beta[0].data.cpu().numpy())
    beta_code = [np.argmax(beta.data.cpu().numpy(), axis=0) for beta in meta_optim.beta]
    print('Greedy beta code: {}'.format(beta_code))


#'''
''' Train with traditional optimizer
print('Traditional Optimizer: SGD')
model = Net().to(device)
direct_optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        direct_optim.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
'''
''' Train with other's implementation of direct gradient meta optimizer
meta_model = Net().to(device)
meta_optimizer = meta_gbg.MetaOptimizer(meta_gbg.MetaModel(meta_model), 2, 10).to(device)
optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)
model = Net().to(device)
loss_sum = 0.0
running_loss, running_new_loss = 0.0, 0.0
prev_loss = torch.zeros(1).cuda()
meta_optimizer.reset_lstm(keep_states=False, model=model, use_cuda=True)
for epoch in range(2):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        meta_model = meta_optimizer.meta_update(model, loss.data, DoOverwrite=(i % update_step == 0))
        outputs = meta_model(inputs)
        new_loss = criterion(outputs, labels)
        #new_loss.backward(retain_graph=True)
        loss_sum += (new_loss - Variable(prev_loss))
        prev_loss = new_loss.data
        if(i % bptt_step == 0):
            meta_optimizer.zero_grad()
            loss_sum.backward()
            for param in meta_optimizer.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            meta_optimizer.reset_lstm(
                    keep_states=i > 0, model=model, use_cuda=True)
            loss_sum = 0.0
        running_loss += loss.item()
        running_new_loss += new_loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f, new loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000, running_new_loss/2000))
            running_loss = 0.0
            running_new_loss = 0.0
'''
print('Finished Training')
eval(model)
#eval(meta_model)
'''
# use trained beta to train new model
second_model = Net().to(device)
meta_optim = MetaOptimizer(second_model.parameters())
meta_helper.reset(second_model)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        second_model.zero_grad()

        # forward + backward + optimize
        outputs = second_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        params_updates = meta_optim.step()
        second_model.zero_grad()
        meta_helper.meta_update(second_model, params_updates, DoUpdate=((i+1) % update_step == 0))
        meta_helper.reset(second_model)
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
eval(second_model)
'''
