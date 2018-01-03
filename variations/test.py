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

num_classes = 35

#transforms defined for train,test
data_transform1 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
data_transform2 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


#data loading code
def dataloader(root,transform,shuffle):
	data = datasets.ImageFolder(root=root,transform=transform)
	return torch.utils.data.DataLoader(data,batch_size=128,shuffle=shuffle,num_workers=8)

trainloader = dataloader('dataset/train',data_transform1,True)
valloader = dataloader('dataset/validation',data_transform2,False)
testloader = dataloader('dataset/test',data_transform2,False)


#defining the model
class Alex_Net(nn.Module):
	def __init__(self, num_classes = 35):
		super(Alex_Net, self).__init__()
		self.convolutionlayers = nn.Sequential(
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
		self.linearlayers = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(128, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, num_classes),
		)
	def forward(self,x):
		x = self.convolutionlayers(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		x = self.linearlayers(x)
		return x




def train(model,model_optimal):
	model = model.cuda()
	model_optimal = model_optimal.cuda()
	loss_func = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01,weight_decay = 0.0005, momentum=0.9)
	scheduler = ls.ReduceLROnPlateau(optimizer,  mode='max', factor=0.1, patience=2,verbose=True,min_lr=1e-5)
	best_acc = 0.0
	end = 15
	epoch = 0
	iter_without_inc = 0
	train_error = []
	iter_train = []

	while iter_without_inc < end:
		model.train()
		epoch = epoch+1
		running_loss = 0.0

		#bachpropagation for a epoch
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = loss_func(outputs, labels)
			loss.backward()
			optimizer.step()

		#calculating accuracy on validation and train set
		validation_acc = tester(valloader,model)
		print('Accuracy on validation set: %f %%' % (validation_acc))
		training_acc = tester(trainloader,model)
		train_error.append(training_acc)
		iter_train.append(epoch)
		scheduler.step(validation_acc)

		#saving the best model 
		if validation_acc > best_acc:
			best_acc = validation_acc
			iter_without_inc = 0
			model_optimal.load_state_dict(model.state_dict())
		else:
			iter_without_inc = iter_without_inc +1

	#printing traing error per epoch
	print('trainng error')
	for i in range(epoch):
		print(iter_train[i],train_error[i])
	return model_optimal

def tester(test_loader,model_optimal):
	model_optimal.eval()
	correct_label = 0
	total_entry = 0
	for data in test_loader:
		images, labels = data
		images, labels = Variable(images.cuda()), labels.cuda()
		outputs = model_optimal(images)
		_, predicted = torch.max(outputs.data, 1)
		total_entry += labels.size(0)
		correct_label += (predicted == labels).sum()

	return (100*correct_label/total_entry)

#defining two instance of our model
alex_net = Alex_Net()
alex_net_optimal = Alex_Net()


start_time = time.time()
model_optimal = train(alex_net,alex_net_optimal)
print('Train time :- %f ' % (time.time()-start_time))
start_time = time.time()
validation_acc = tester(valloader,model_optimal)
print('test time for each example:- %f' %((time.time()-start_time)/(len(valloader)*128)))
print('Accuracy on validation set: %f %%' % (
	validation_acc))
test_acc = tester(testloader,model_optimal)
print('Accuracy on test set: %f %%' % (
	test_acc))
torch.save(model_optimal.state_dict(),'./model128.pth')
