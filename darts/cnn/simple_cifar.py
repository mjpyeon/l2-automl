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


class MetaHelper(nn.Module):
    def __init__(self, model):
        self.meta_model = model
    def reset(self, model):
        self.meta_model.detach()
        self.meta_model.model.zero_grad()
        self.meta_model.copy_params_from(model)
    def meta_update(self, updates):
        # Meta update itself
        flat_params = self.meta_model.get_flat_params()
        flat_params += torch.cat([up.view(-1) for up in updates])
        flat_params = flat_params.view(-1)
        self.meta_model.set_flat_params(flat_params)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
meta_helper = MetaHelper(MetaModel(Net().to(device)))
model = Net().to(device)

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
criterion = nn.CrossEntropyLoss()
meta_optim = MetaOptimizer(model.parameters())
optimizer = optim.SGD(meta_optim.beta, lr=0.001)
#new_params = [p.clone() for p in net_params]
#pdb.set_trace()
bptt_step = 1
update_step = 5
print('bptt_step: {}, update_step: {}'.format(bptt_step, update_step))
#prev_loss = torch.zeros(1).cuda()
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    running_new_loss = 0.0
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
        params_updates = meta_optim.step(DoUpdate=(i % update_step == 0))
        meta_model = meta_helper.meta_update(params_updates)
        outputs = meta_model(inputs)
        #assert len(params_updates) == len(net_params)
        #for i in range(len(params_updates)):
        #   new_params[i] += params_updates[i]

        new_loss = criterion(outputs, labels)
        new_loss.backward(retain_graph=True)
        optimizer.step()
        #prev_loss = new_loss.data
        if(i % bptt_step == 0):
            meta_helper.reset(model)
            optimizer.zero_grad()
            #prev_loss = torch.zeros(1).cuda()

        # print statistics
        running_loss += loss.item()
        running_new_loss += new_loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f, new loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000, running_new_loss/2000))
            running_loss = 0.0
            running_new_loss = 0.0
        if i % 5000 == 0:
            print(meta_optim.beta)

print('Finished Training')
eval(model)
#eval(meta_model)

