import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt

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

# data_transform2 = transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop(256),
#         transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

dataset1 = datasets.ImageFolder(root='~/dataset/train',
                                           transform=data_transform1)

# dataset2 = datasets.ImageFolder(root='dataset/train',
#                                            transform=data_transform2)

# dataset = torch.utils.data.ConcatDataset([dataset1,dataset2])

dataset = dataset1

#Edit here
val_dataset = datasets.ImageFolder(root='~/dataset/validation',
                                           transform=data_transform1)

test_dataset = datasets.ImageFolder(root='~/dataset/test',
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
		init.normal(self.convolution[0].weight, mean=0, std=0.01)
		init.constant(self.convolution[0].bias, 0)
		init.normal(self.convolution[1].weight, mean=0, std=0.01)
		init.constant(self.convolution[1].bias, 1)
		init.normal(self.convolution[2].weight, mean=0, std=0.01)
		init.constant(self.convolution[2].bias, 0)
		init.normal(self.convolution[3].weight, mean=0, std=0.01)
		init.constant(self.convolution[3].bias, 1)
		init.normal(self.convolution[4].weight, mean=0, std=0.01)
		init.constant(self.convolution[4].bias, 1)

		self.linear = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)
		init.normal(self.linear[0].weight, mean=0, std=0.01)
		init.constant(self.linear[0].bias, 1)
		init.normal(self.linear[1].weight, mean=0, std=0.01)
		init.constant(self.linear[1].bias, 1)
		init.normal(self.linear[2].weight, mean=0, std=0.01)
		init.constant(self.linear[2].bias, 1)

	def forward(self,x):
		x = self.convolution(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.linear(x)
		return x

def train():
	alex_net = AlexNet()
	alex_net = alex_net.cuda()
	alex_net_optimal = AlexNet()
	alex_net_optimal = alex_net_optimal.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(alex_net.parameters(), lr=0.01,wd = 0.0005, momentum=0.9)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min', factor=0.1, patience=10)

	#stopping criteria parameters
	wait = 0
	best_loss = 1e15
	min_delta = 1e-5
	p = 10
	#for epoch in range(2):  # loop over the dataset multiple times

	epoch = 0
	j = 0

	train_error = []
	iter_train = []
	while epoch < p:
		j = j+1
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs, labels = data

			# wrap them in Variable
			inputs, labels = Variable(inputs), Variable(labels)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = alex_net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# # print statistics
			running_loss += loss.data[0]
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0


		train_error.append(running_loss/trainloader._len_())
		iter_train.append(j)

		val_input,val_label = val_dataset
		val_output = alex_net(val_input)
		val_loss = criterion(val_output,val_label)
		scheduler.step(val_loss)  

		if (val_loss.data[0] - best_loss) < -min_delta:
			best_loss = val_loss.data[0]
			epoch = 0
			alex_net_optimal.load_state_dict(alex_net.state_dict())
		else:
			epoch = epoch +1

	#print('Finished Training')
	plt.plot(iter_train, train_error, label='Train')
	plt.xlabel('Epoch')
	plt.ylabel('Cross-Entropy Loss')
	plt.legend()

	return alex_net_optimal

def test(test_loader,model):
	correct = 0
	total = 0
	for data in test_loader:
		images, labels = data
		outputs = model(Variable(images))
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	# print('Accuracy of the network on the validation set: %d %%' % (
 #    	100 * correct / total))
	return (100*correct/total)


if __name__ == '__main__':
	model = train()
	acc = test(valloader,model)
	print('Accuracy of the network on the validation set: %d %%' % (
    	acc))
	acc = test(testloader,model)
	print('Accuracy of the network on the test set: %d %%' % (
    	acc))

