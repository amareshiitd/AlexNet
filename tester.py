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
import os
import shutil
# /home/amareshiitd/Desktop/deep learning/assn1/
default = ''

label=['Apple','Baby','Bicycle','Bird','Bus','Camel','Car','Chimpanzee','Clock','Crocodile','Deer','Dog','Elephant','Fish','Flower','Frog','Guitar','Horse','Lamp','Lock','Man','Motorcycle','Mushroom','Orange','Pear','Plane','Rocket','Ship','Table','Television','Tiger','Tractor','Train','Truck','Woman']
print(len(label))
for i in range(35):
	print(i)
	for j in ['test']:
		src = default+'ImageNet_Subset/'+label[i]+'/'+j
		src_files = os.listdir(src)
		directory = default+'testdataset/'+j+'/'+label[i]+'/'
		os.makedirs(directory)
		for file_name in src_files:
		    full_file_name = os.path.join(src, file_name)
		    dest = default+'testdataset/'+j+'/'+label[i]+'/'+file_name
		    if (os.path.isfile(full_file_name)):
		        shutil.copy(full_file_name, dest)


num_classes = 35
	
data_transform2 = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
test_dataset = datasets.ImageFolder(root='testdataset/test',transform=data_transform2)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=128, shuffle=False,num_workers=8)


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



alex_net = Alex_Net()


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
		correct += (predicted == labels).sum()
	return (100*correct/total)


if __name__ == '__main__':
	model = alex_net.cuda()
	model.load_state_dict(torch.load('model128.pth'))
	acc = test(testloader,model)
	print('Accuracy of the network on the test set: %f %%' % (
    	acc))
