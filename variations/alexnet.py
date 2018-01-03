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

# import matplotlib.pyplot as plt

num_classes = 35
	
data_transform1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# data_transform2 = transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop(256),
#         transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

dataset1 = datasets.ImageFolder(root='dataset/train',
                                           transform=data_transform1)

# dataset2 = datasets.ImageFolder(root='dataset/train',
#                                            transform=data_transform2)

# dataset = torch.utils.data.ConcatDataset([dataset1,dataset2])

dataset = dataset1

#Edit here
val_dataset = datasets.ImageFolder(root='dataset/validation',
                                           transform=data_transform1)

test_dataset = datasets.ImageFolder(root='dataset/test',
                                           transform=data_transform1)

trainloader = torch.utils.data.DataLoader(dataset,batch_size=128, shuffle=True,
                                             num_workers=4)

valloader = torch.utils.data.DataLoader(val_dataset,batch_size=128, shuffle=True,
                                             num_workers=4)

testloader = torch.utils.data.DataLoader(test_dataset,batch_size=128, shuffle=True,
                                             num_workers=4)


class AlexNet(nn.Module):
	def __init__(self, num_classes = 35):
		super(AlexNet, self).__init__()
		self.convolution = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(96, 256, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		# init.normal(self.convolution[0].weight, mean=0, std=0.01)
		# init.constant(self.convolution[0].bias, 0)
		# init.normal(self.convolution[3].weight, mean=0, std=0.01)
		# init.constant(self.convolution[3].bias, 1)
		# init.normal(self.convolution[6].weight, mean=0, std=0.01)
		# init.constant(self.convolution[6].bias, 0)
		# init.normal(self.convolution[8].weight, mean=0, std=0.01)
		# init.constant(self.convolution[8].bias, 1)
		# init.normal(self.convolution[10].weight, mean=0, std=0.01)
		# init.constant(self.convolution[10].bias, 1)

		self.linear = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)
		# init.normal(self.linear[1].weight, mean=0, std=0.01)
		# init.constant(self.linear[1].bias, 1)
		# init.normal(self.linear[4].weight, mean=0, std=0.01)
		# init.constant(self.linear[4].bias, 1)
		# init.normal(self.linear[6].weight, mean=0, std=0.01)
		# init.constant(self.linear[6].bias, 1)

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
	scheduler = ls.ReduceLROnPlateau(optimizer,  mode='max', factor=0.1, patience=2,verbose=True)

	#stopping criteria parameters
	# wait = 0
	best_acc = 0.0
	min_delta = 1e-3
	p = 5

	epoch = 0
	j = 0

	train_error = []
	iter_train = []
	while epoch < p:
		model.train()
		j = j+1
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data

			# wrap them in Variable
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# # print statistics
			running_loss += loss.data[0]
			# if i % 2000 == 0:    # print every 2000 mini-batches
			# 	print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
			# 	running_loss = 0.0


		# train_error.append(test(trainloader,alex_net))
		# iter_train.append(j)
		# acc = test(valloader,alex_net)
		# val_acc=0.0
		# val_loss = 0.0
		# correct = 0
		# total = 0
		# for i, data in enumerate(valloader, 0):
		# 	val_input,val_label = data
		# 	val_input, val_label = Variable(val_input.cuda()), val_label.cuda()
		# 	val_output = alex_net(val_input)
		# 	loss = criterion(val_output,Variable(val_label))
		# 	val_loss+=loss.data[0]
		# 	_, predicted = torch.max(val_output.data, 1)
		# 	total += val_label.size(0)
		# 	correct += (predicted == val_label).sum()
			
		val_acc = test(valloader,model)
		print('Accuracy of the network on the validation set: %.5f %%' % (val_acc))
		scheduler.step(val_acc)  
		if (val_acc - best_acc) > min_delta:
			best_acc = val_acc
			epoch = 0
			model_optimal.load_state_dict(model.state_dict())
			# alex_net_optimal = copy.deepcopy(alex_net)
		else:
			epoch = epoch +1

	#print('Finished Training')
	# plt.plot(iter_train, train_error, label='Train')
	# plt.xlabel('Epoch')
	# plt.ylabel('Cross-Entropy Loss')
	# plt.legend()
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
	model = train(alex_net,alex_net_optimal)
	acc = test(valloader,model)
	print('Accuracy of the network on the validation set: %f %%' % (
    	acc))
	acc = test(testloader,model)
	print('Accuracy of the network on the test set: %f %%' % (
    	acc))
	# torch.save(model.state_dict(),'./base_model.pth')