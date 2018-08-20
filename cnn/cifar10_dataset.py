import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from args import args

class Cifar10:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
            download=True, transform=self.transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
            download=True, transform=self.transform)
        self.classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 
            'horse', 'ship', 'truck')
        self.num_classes = len(self.classes)
        self.num_train_batch = (len(self.train_set) - 1) / args.batch_size + 1
        self._build_loaders()

    def _build_loaders(self):
        indices = list(range(len(self.train_set)))
        split = int(np.floor(args.train_portion * len(indices)))
        self.train_loader = torch.utils.data.DataLoader(
              self.train_set, batch_size=args.batch_size,
              #sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
              shuffle=True,
              pin_memory=True, num_workers=0)
        self.val_loader = torch.utils.data.DataLoader(
              self.train_set, batch_size=args.batch_size,
              #sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:len(self.train_set)]),
              shuffle=True,
              pin_memory=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(
              self.test_set, batch_size=args.batch_size, shuffle=False,
              pin_memory=True, num_workers=0)
    
