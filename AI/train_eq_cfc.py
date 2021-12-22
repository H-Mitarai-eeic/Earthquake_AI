
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import MyDataSet
from myloss import MyLoss
from myloss import MyLoss2
from mycfc import MYFCN


import csv

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: CIFAR-10')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=20,
						help='Number of sweeps over the training data')
	parser.add_argument('--frequency', '-f', type=int, default=-1,
						help='Frequency of taking a snapshot')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='result',
						help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	parser.add_argument('--dataset', '-d', default='data100/',
						help='Root directory of dataset')
	parser.add_argument('--mask', '-mask', default='ObservationPointsMap_honshu6464.csv',
						help='Root directory of dataset')
	args = parser.parse_args()
	print("train_eq_fcn")
	print("output: " ,args.out)
	print("dataset: ", args.dataset)
	print("mask: ", args.mask)
	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')

	# Set up a neural network to train
	mesh_size = (64, 64, 10)
	data_channels = 1
	depth_max = 800
	lr = 0.1
	weight = (0.0, 0.0, 1.0)
	#weight = (0.1,)*10
	exponent = 4
	kernel_size = 2
	stride = None
	print("mesh_size: ", mesh_size)
	print("data_channels", data_channels)
	print("depth_max", depth_max)
	print("learning rate: ", lr)
	print("weight: ", weight)
	print("exponent: ", exponent)
	print("kernel_size: ", kernel_size)
	print("stride: ", stride)
	print('')
	net = MYFCN(mesh_size=mesh_size, in_channels=data_channels)

	# Setup a loss and an optimizer
	criterion = MyLoss(kernel_size=kernel_size, stride=stride)
	#criterion = MyLoss2(kernel_size=kernel_size, stride=stride, gpu=args.gpu)
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])
	trainvalset = MyDataSet(channels=data_channels, root=args.dataset, train=True, transform=transform, mesh_size=mesh_size, depth_max=depth_max)

	# Load designated network weight
	if args.resume:
		net.load_state_dict(torch.load(args.resume))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)

	#open mask
	with open(args.mask, "r") as f_mask:
		reader = csv.reader(f_mask)
		mask = [[int(row2) for row2 in row] for row in reader]

	# Split train/val
	n_samples = len(trainvalset)
	print("n_samples:", n_samples)
	trainsize = int(n_samples * 0.9)
	valsize = n_samples - trainsize
	trainset, valset = torch.utils.data.random_split(trainvalset, [trainsize, valsize])

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize,
											  shuffle=True, num_workers=2)
	valloader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize,
											shuffle=True, num_workers=2)
	

	# Setup result holder
	x = []
	ac_train = []
	ac_val = []
	# Train
	for ep in range(args.epoch):  # Loop over the dataset multiple times

		running_loss = 0.0
		loss_train = 0
		total_train = 0
		loss_val = 0
		total_val = 0

		for s, data in enumerate(trainloader, 0):
			# Get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			print(inputs.size())
			#targetsの生成
			targets = torch.zeros(len(labels), len(labels[0]), len(labels[0][0]))

			for B in range(len(labels)):
				for Y in range(len(labels[B])):
					for X in range(len(labels[B][Y])):
						targets[B][Y][X] = labels[B][Y][X]
			
			#maskを生成
			mask_tensor = torch.zeros(len(labels), len(labels[0]), len(labels[0][0]))
			
			for B in range(len(labels)):
				for Y in range(len(labels[B])):
					for X in range(len(labels[B][Y])):
						if mask[Y][X] != 0:
						#if labels[B][Y][X] > 0:
							mask_tensor[B][Y][X] = 1 
		
			if args.gpu >= 0:
				inputs = inputs.to(device)
				labels = labels.to(device)
				targets = targets.to(device)
				mask_tensor = mask_tensor.to(device)
			# Reset the parameter gradients
			optimizer.zero_grad()

			# Forward
			inputs.requires_grad = True
			outputs = net(inputs)

			# Predict the label
				#print("")
				#print("outputs:", outputs.size())
				#print("outputs:", outputs.size())
				#print("targets:", targets.size())
				#print("labels:", labels.size())
				#print("mask_tensor:", mask_tensor.size())

			"""
				print("labels:",labels.size())
				print("outputs:",outputs.size())
			"""

			# Backward + Optimize
			loss = criterion(outputs=outputs, targets=targets, mask=mask_tensor, weight=weight, exponent=exponent)
			#print(loss)
			loss.backward()
			optimizer.step()

			# Add loss
			running_loss += loss.item()
			print("trainloader_",s,"loss:",loss.item())
			# record loss for graph
			loss_train += loss.item()
			total_train += 1

		# Report loss of the epoch
		print('[epoch %d] loss: %.3f' % (ep + 1, running_loss))

		# Save the model
		if (ep + 1) % args.frequency == 0:
			path = args.out + "/model_" + str(ep + 1)
			torch.save(net.state_dict(), path)

		# Validation
		with torch.no_grad():
			for data in valloader:
				images, labels = data
				#targetsの生成
				targets = torch.zeros(len(labels), len(labels[0]), len(labels[0][0]))
				for B in range(len(labels)):
					for Y in range(len(labels[B])):
						for X in range(len(labels[B][Y])):
							targets[B][Y][X] = labels[B][Y][X]
				#maskを生成
				mask_tensor = torch.zeros(len(labels), len(labels[0]), len(labels[0][0]))
				for B in range(len(labels)):
					for Y in range(len(labels[B])):
						for X in range(len(labels[B][Y])):
							if mask[Y][X] != 0:
								mask_tensor[B][Y][X] = 1 

				if args.gpu >= 0:
					images = images.to(device)
					labels = labels.to(device)
					targets = targets.to(device)
					mask_tensor = mask_tensor.to(device)

				outputs = net(images)
				loss = criterion(outputs=outputs, targets=targets, mask=mask_tensor, weight=weight, exponent=exponent)
				print("validation loss: ", loss.item())
				#record loss for drawing graph
				loss_val += loss.item()
				total_val += 1

		# Record result
			print('')
			x.append(ep + 1)
			ac_train.append(loss_train / total_train)
			ac_val.append(loss_val / total_val)


	print('Finished Training')
	path = args.out + "/model_final"
	torch.save(net.state_dict(), path)

	# Draw graph
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(x, ac_train, label='Training')
	ax.plot(x, ac_val, label='Validation')
	ax.legend()
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy [%]")
	ax.set_ylim(0, max(ac_val))

	plt.savefig(args.out + '/accuracy_cifar.png')
	#plt.show()

if __name__ == '__main__':
	main()