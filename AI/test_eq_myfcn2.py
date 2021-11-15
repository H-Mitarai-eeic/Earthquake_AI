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
from myfcn2 import MYFCN2

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: CIFAR-10')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
	parser.add_argument('--dataset', '-d', default='data/mini_cifar',
						help='Root directory of dataset')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('')

	# Set up a neural network to test
	net = MYFCN2(10)
	# Load designated network weight
	net.load_state_dict(torch.load(args.model))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		print("GPU using")
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])
	testset = MyDataSet(root=args.dataset, train=False, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
											 shuffle=False, num_workers=2)

	# Test
	correct = 0
	total = 0
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			# Get the inputs; data is a list of [inputs, labels]
			images, labels = data
			if args.gpu >= 0:
				images = images.to(device)
				labels = labels.to(device)
			# Forward
			outputs = net(images)
			# Predict the label
			# for i in range(0,255,20):
			# 	for j in range(0,255,20):
			# 		print("[{}, {}]".format(i,j), outputs[0, :, i, j])

			_, predicted = torch.max(outputs, 1)
			# Check whether estimation is right
			c = (predicted == labels).squeeze()
			for i in range(len(predicted)):
				for j in range(len(predicted[i])):
					for k in range(len(predicted[i][j])):
						label = labels[i][j][k]
						correct += c[i][j][k].item()
						class_correct[label] += c[i][j][k].item()
						total += 1
						class_total[label] += 1

	# List of classes
	classes = ("0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7")
	# Show accuracy
	for i in range(10):
		if class_total[i] != 0:
			print('Accuracy of %5s : %2d %% , total num of this class: %d' % (
			classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))
	print('Accuracy : %.3f %%' % (100 * correct / total))


if __name__ == '__main__':
	main()
