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

import csv

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
	parser.add_argument('--mask', '-mask', default='ObservationPointsMap.csv',
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
	net = FCN32s(10)
	# Load designated network weight
	net.load_state_dict(torch.load(args.model))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])
	testset = MyDataSet(root=args.dataset, train=False, transform=transform, ID=args.ID)
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
			# Forward
			outputs = net(images)
			# Predict the label
			_, predicted = torch.max(outputs, 1)
			# Check whether estimation is right
			c = (predicted == labels).squeeze()
			'''
			print(len(output))
			print(len(output[0]))
			print(len(output)[0][0])
			print(len(predicted))
			print(len(predicted[0]))
			print(len(predicted)[0][0])
			'''
			print(outputs.size())
			print(predicted.size())
			"""
			for i in range(len(predicted)):
				for j in range(len(predicted[i])):
					for k in range(len(predicted[i][j])):
						label = labels[i][j][k]
						correct += c[i][j][k].item()
						class_correct[label] += c[i][j][k].item()
						total += 1
						class_total[label] += 1
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


	# List of classes
	classes = ("0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7")
	classes_diff_ver = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
	"""
	# Show accuracy
	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
	print('Accuracy : %.3f %%' % (100 * correct / total))
	"""
	# Show accuracy
	print("予測震度と実際の震度のずれの分布")
	for i in range(10):
		print('%5s 階級 : %2d %%' % (classes_diff_ver[i], 100 * class_diff[i] / total))


	#csv出力
	predicted_map = predicted.clone().squeeze().tolist()
	#print(predicted_map)
	with open(args.output + 'predicted/' + args.ID + '_predicted.csv', "w") as fo:
		writer = csv.writer(fo)
		writer.writerows(predicted_map)

if __name__ == '__main__':
	main()
