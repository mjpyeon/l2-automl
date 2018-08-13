import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from meta_optimizer import MetaOptimizer
from cifar10_dataset import Cifar10
from args import args

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

def train_from_scratch(net, dataset, optimizer = 'SGD', beta = None):
	model = net().to(device)
	if optimizer is 'SGD':
		optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
	elif optimizer is 'meta':
		assert(beta is not None)
		optim = MetaOptimizer(model.parameters(), beta, DoUpdate=True)
	for epoch in range(args.epoch):
	    running_loss = 0.0
	    for i, data in enumerate(dataset.train_loader, 0):
			x_train, y_train = data[0].to(device), data[1].to(device)
			model.zero_grad()
			outputs = model(x_train)
			loss = criterion(outputs, y_train)
	        loss.backward()
			# TODO(Hao): why do this?
			#grad_norm = nn.utils.clip_grad_norm_(second_model.parameters(), args.model_grad_norm)
	        #for param in model.parameters():
	        #    param.grad = param.grad * torch.sqrt(torch.abs(param.grad))
	        optim.step()
	        running_loss += loss.item()
	        if (i+1) % args.log_freq == 0:    # print every 2000 mini-batches
	            print('[%d, %5d] loss: %.3f' %
	                  (epoch + 1, i + 1, running_loss / args.log_freq))
	            running_loss = 0.0
	    if (epoch+1) % args.test_freq == 0:
	        print('Run Testing')
	        dataset.eval(model)
