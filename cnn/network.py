import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		pass
	
	def _make_layers(self):
		pass
	
	def forward(self, x):
		raise NotImplementedError
	
	def update(self, updates):
		"""
		non-differentiable update
		"""
		for param, update in zip(self.parameters(), updates):
			param.data.add_(update.data)
	
	def differentiable_update(self, updates):
		"""
		differentiable update, i.e. we assign a differentiable variable as the model parameters of a NN, which
		is disallowed by PyTorch actually. The way we do this is to first create a new variable `updated_param` 
		to store the updated result and then assign it to the model

		There is no better way to reassign the parameter as a new variable than directly accessing _parameters
		"""
		self.count = 0
		def _iter_child(module):
			for child in module.children():
				if len(child._parameters) > 0:
					for key, param in child._parameters.items():
						if param is not None:
							updated_params = child._parameters[key] + updates[self.count]
							child._parameters[key] = updated_params 
							self.count += 1
				elif len(child._modules) > 0:
					_iter_child(child)
		_iter_child(self)
	
	def detach(self):
		"""
		This detaches the model.parameters from the past path that generates it.
		"""
		def _iter_child(module):
			for child in module.children():
				if len(child._parameters) > 0:
					for key, param in child._parameters.items():
						if param is not None:
							child._parameters[key] = param.detach()
				elif len(child._modules) > 0:
					_iter_child(child)
		_iter_child(self)

	def copy_params_from(self, source):
		"""
		non-differentiable copy
		"""
		for dest, src in zip(self.parameters(), source.parameters()):
			dest.data.copy_(src.data)
	
	def copy_params_to(self, target):
		"""
		non-differentiable copy
		"""
		for src, dest in zip(self.parameters(), target.parameters()):
			dest.data.copy_(src.data)

class Simple_Net(Network):
    def __init__(self):
        super(Simple_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.history = {}
    def forward(self, x, Eval=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGG(Network):
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(VGG.cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(Network):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in MobileNetV2.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test_mobilenet():
    net = MobileNetV2()
    flat_params = net.get_flat_params()
    print(flat_params)
    net2= MobileNetV2()
    v1 = net2.get_flat_params()
    print(v1)
    net2.set_flat_params(flat_params)
    v2 = net2.get_flat_params()
    print(v2)

def test_vgg():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

#test_mobilenet()
#test_vgg()
