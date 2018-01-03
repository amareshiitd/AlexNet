from __future__ import division
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import lr_scheduler as ls
import copy
import time
# import matplotlib.pyplot as plt

num_classes = 35
	
data_transform1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
data_transform2 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
dataset = datasets.ImageFolder(root='dataset/train',transform=data_transform1)
val_dataset = datasets.ImageFolder(root='dataset/validation',transform=data_transform2)
test_dataset = datasets.ImageFolder(root='dataset/test',transform=data_transform2)
trainloader = torch.utils.data.DataLoader(dataset,batch_size=128, shuffle=True,num_workers=8)
valloader = torch.utils.data.DataLoader(val_dataset,batch_size=128, shuffle=False,num_workers=8)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=128, shuffle=False,num_workers=8)


class AlexNet(nn.Module):
	def __init__(self, num_classes = 35):
		super(AlexNet, self).__init__()
		self.convolution = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(96, 256, kernel_size=5, padding=2),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.Tanh(),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.Tanh(),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.linear = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.Tanh(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.Tanh(),
			nn.Linear(4096, num_classes),
		)
	def forward(self,x):
		x = self.convolution(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.linear(x)
		return x


alex_net = AlexNet()
alex_net_optimal = AlexNet()

def train(model,model_optimal):
	model = model.cuda()
	model_optimal = model_optimal.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01,weight_decay = 0.0005, momentum=0.9)
	scheduler = ls.ReduceLROnPlateau(optimizer,  mode='max', factor=0.1, patience=2,verbose=True,min_lr=1e-5)

	#stopping criteria parameters
	# wait = 0
	best_acc = 0.0
	p = 10

	epoch = 0
	j = 0

	train_error = []
	iter_train = []
	while epoch < p:
		model.train()
		j = j+1
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data

			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

			optimizer.zero_grad()

			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.data[0]
			
		val_acc = test(valloader,model)
		print('Accuracy of the network on the validation set: %.5f %%' % (val_acc))
		acc = test(trainloader,model)
		train_error.append(acc)
		iter_train.append(j)
		scheduler.step(val_acc)  
		if val_acc > best_acc:
			best_acc = val_acc
			epoch = 0
			model_optimal.load_state_dict(model.state_dict())
		else:
			epoch = epoch +1
	print('trainng error')
	for i in range(j):
		print(iter_train[i],train_error[i])
	return model_optimal

def test(test_loader,model):
	model.eval()
	correct = 0
	total = 0
	for data in test_loader:
		images, labels = data
		images, labels = Variable(images.cuda()), labels.cuda()
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		# print(predicted)

		# print(labels)
		# print(type(predicted))
		# print(type(labels))
		correct += (predicted == labels).sum()

	# print('Accuracy of the network on the validation set: %d %%' % (
 #    	100 * correct / total))
	return (100*correct/total)


if __name__ == '__main__':
	start_time = time.time()
	model = train(alex_net,alex_net_optimal)
	print('Train time :- %f ' % (time.time()-start_time))
	start_time = time.time()
	acc = test(valloader,model)
	print('test time for each example:- %f' %((time.time()-start_time)/(len(valloader)*128)))
	print('Accuracy of the network on the validation set: %f %%' % (
    	acc))
	acc = test(testloader,model)
	print('Accuracy of the network on the test set: %f %%' % (
    	acc))

	# torch.save(model.state_dict(),'./base_model.pth')