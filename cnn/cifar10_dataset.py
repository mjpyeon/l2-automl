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
        self._build_loaders()

    def _build_loaders(self):
        indices = list(range(len(self.train_set)))
        split = int(np.floor(args.train_portion * len(indices)))
        self.train_loader = torch.utils.data.DataLoader(
              self.train_set, batch_size=args.batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
              pin_memory=True, num_workers=1)
        self.val_loader = torch.utils.data.DataLoader(
              self.train_set, batch_size=args.batch_size,
              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:len(self.train_set)]),
              pin_memory=True, num_workers=1)
        self.test_loader = torch.utils.data.DataLoader(
              self.test_set, batch_size=args.batch_size, shuffle=False,
              pin_memory=True, num_workers=2)
    
    def eval(self, model):
        correct, total = 0, 0
        with torch.no_grad():
            for image, labels in self.test_loader:
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
