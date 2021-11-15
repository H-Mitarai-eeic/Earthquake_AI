import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fcn8s import FCN8s
from fcn32s import FCN32s
from dataset import MyDataSet
from myfcn import MYFCN
from myfcn import MYFCN2
from myfcn import MYFCN3

import csv
import copy

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: CIFAR-10')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	parser.add_argument('--dataset', '-d', default='data100/',
						help='Root directory of dataset')
	parser.add_argument('--output', '-o', default='data100/',
						help='Root directory of outputfile')					
	parser.add_argument('--ID', '-i', default='40278',
						help='Erthquake ID for input/output files')
	parser.add_argument('--mask', '-mask', default='ObservationPointsMap_honshu6464.csv',
						help='Root directory of dataset')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('')

	#open mask
	with open(args.mask, "r") as f_mask:
		reader = csv.reader(f_mask)
		mask = [[int(row2) for row2 in row] for row in reader]

	# Set up a neural network to test
	data_channels = 3
	#net = FCN32s(10)
	#net = MYFCN(10)
	#net = MYFCN2(in_channels=data_channels, n_class=10)
	net = MYFCN3(in_channels=data_channels, n_class=10)
	# Load designated network weight
	net.load_state_dict(torch.load(args.model))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])
	testset = MyDataSet(chanels=data_channels, root=args.dataset, train=False, transform=transform, ID=args.ID)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
											 shuffle=False, num_workers=2)

	# Test
	correct = 0
	total = 0
	class_correct = list(0. for i in range(10))
	class_diff = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		# 多分1つしかテストしないようになっているはず
		for data in testloader:
			# Get the inputs; data is a list of [inputs, labels]
			images, labels = data
			if args.gpu >= 0:
				images = images.to(device)
				labels = labels.to(device)
			#images成形
			print("images:", images.size())
			print("mask:", len(mask))
			if data_channels == 3:
				for B in range(len(images)):
					for X in range(len(images[B][0])):
						for Y in range(len(images[B][0][X])):
							images[B][2][X][Y] = mask[X][Y]
			for B in range(len(images)):
				for X in range(len(images[B][1])):
					for Y in range(len(images[B][1][X])):
						if mask[X][Y] == 0:
							images[B][1][X][Y] = 0
			# Forward
			outputs = net(images)
			# Predict the label
			#_, predicted = torch.max(outputs, 1)
			predicted = [[0 for i in range(len(labels[0][0]))] for j in range(len(labels[0]))]
			print("output:", outputs.size())
			
			for B in range(len(outputs)):
				for C in range(len(outputs[B])):
					for X in range(len(outputs[B][C])):
						for Y in range(len(outputs[B][C][X])):
							if outputs[B][C][X][Y].item() > 0:
								predicted[X][Y] = C
			
			# Check whether estimation is right
			#c = (predicted == labels).squeeze()

			print("outputs : ", outputs.size())
			#print(predicted.size())
			"""
			for i in range(len(predicted)):
				for j in range(len(predicted[i])):
					for k in range(len(predicted[i][j])):
						if mask[j][k] != 0:
							label = labels[i][j][k]
							predic = predicted[i][j][k]
							class_diff_index = int(abs(label - predic))
							class_diff[class_diff_index] += 1
							total += 1 
			"""
			for B in range(len(labels)):
				for X in range(len(labels[B])):
					for Y in range(len(labels[B][X])):
						if mask[X][Y] != 0:
							label = labels[B][X][Y]
							predic = predicted[X][Y]
							class_diff_index = int(abs(label - predic))
							class_diff[class_diff_index] += 1
							total += 1 
			

	# List of classes
	classes = ("0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7")
	classes_diff_ver = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
	
	# Show accuracy
	"""	for i in range(10):
			print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
		print('Accuracy : %.3f %%' % (100 * correct / total))
	"""
	# Show accuracy
	print("予測震度と実際の震度のずれの分布")
	for i in range(10):
		print('%5s 階級 : %2d %%' % (classes_diff_ver[i], 100 * class_diff[i] / total))


	#csv出力
	#predicted_map = predicted.clone().squeeze().tolist()
	predicted_map = copy.deepcopy(predicted)
	for X in range(len(predicted_map)):
		for Y in range(len(predicted_map[X])):
			if mask[X][Y] == 0:
				#predicted_map[X][Y] =0
				predicted_map[X][Y] = predicted_map[X][Y]

	#print(predicted_map)
	with open(args.output + 'predicted/' + args.ID + '_predicted.csv', "w") as fo:
		writer = csv.writer(fo)
		writer.writerows(predicted_map)

	for i in range(10):
		print("class", i ,"at 100 100:", outputs[0][i][10][10])
		print("class", i, "at 50 100:", outputs[0][i][20][10])

if __name__ == '__main__':
	main()
