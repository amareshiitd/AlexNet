import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable
import json
from random import randint
import operator

#reading validation data
val_data = []
val_label = []
file = open("RNN_Data_files/val_sentences.txt", "r")
for line in file:
	val_data.append(line.split())
file = open("RNN_Data_files/val_tags.txt", "r")
for line in file:
	val_label.append(line.split())

with open('word.txt', 'r') as f:
	word_to_ix = json.load(f)

with open('tag.txt', 'r') as f:
	tag_to_ix = json.load(f)

maxi = max(word_to_ix.values())


print(maxi)


def prepare_sequence(seq, to_ix):
	idxs =[]
	for w in seq:
		if w not in to_ix:
			idxs.append(randint(0, maxi))
		else:
			idxs.append(to_ix[w])
	tensor = torch.LongTensor(idxs)
	return tensor.cuda()

def create_data(word_to_ix,tag_to_ix,data,label):
	data1 =[]
	label1 =[]
	for i in range(len(data)):
		data1.append(prepare_sequence(data[i], word_to_ix))
		label1.append(prepare_sequence(label[i], tag_to_ix))
	data = data1
	label = label1
	return data,label

val_data,val_label=create_data(word_to_ix,tag_to_ix,val_data,val_label)


class tagger(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(tagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.GRUCell(embedding_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return autograd.Variable(torch.zeros( 1, self.hidden_dim).cuda())

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		output = []
		for word in embeds:
			self.hidden = self.lstm(word.view(1, -1), self.hidden)
			output.append(self.hidden)
		output = torch.stack(output)
		tag_space = self.hidden2tag(output.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space)
		return tag_scores
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
tag = tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))


def tester(data,label,model):
	total = 0
	correct = 0
	for i in range(len(data)):
		input_data = Variable(data[i])
		input_tag = label[i]
		tag_scores = model(input_data)
		_, predicted = torch.max(tag_scores.data, 1)
		total+= len(input_tag)
		correct += (predicted == input_tag).sum()
	return (100*correct/total)

if __name__ == '__main__':
	model = tag.cuda()
	model.load_state_dict(torch.load('taggergru.pth'))
	acc = tester(val_data,val_label,model)
	print('Accuracy of the network on the test set: %f %%' % (
    	acc))
