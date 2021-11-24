
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from network import CifarCNN
#from fcn8s import FCN8s
#from fcn32s import FCN32s
from dataset import MyDataSet
from myloss import MyLoss
#from myfcn import MYFCN
#from myfcn import MYFCN2
from myfcn import MYFCN4
# from network import EQCNN

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
	data_channels = 2
	lr = 0.1
	print("data_channels: ", data_channels)
	print("learning rate: ", lr)
	print('')
	#net = FCN32s(in_channels=data_channels, n_class=10)
	#net = MYFCN(10)
	#net = MYFCN2(in_channels=data_channels, n_class=10)
	net = MYFCN4(in_channels=data_channels, n_class=10)
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

	# Setup a loss and an optimizer
	#criterion = nn.CrossEntropyLoss()
	criterion = MyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])

	trainvalset = MyDataSet(channels=data_channels, root=args.dataset, train=True, transform=transform, mask=mask)
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
		correct_train = 0
		total_train = 0
		correct_val = 0
		total_val = 0

		for s, data in enumerate(trainloader, 0):
			# Get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			#targetsの生成
			targets = (-1) * torch.ones(len(labels), 10, len(labels[0]), len(labels[0][0]))
			"""
				for B in range(len(labels)):
					for Y in range(len(labels[B])):
						for X in range(len(labels[B][Y])):
							C = labels[B][Y][X]
							if C == 0:
								targets[B][C][Y][X] = 1
							else:
								targets[B][C][Y][X] = 1
			"""
			for B in range(len(labels)):
				for Y in range(len(labels[B])):
					for X in range(len(labels[B][Y])):
						Class = labels[B][Y][X]
						for C in range(0, Class + 1):
							targets[B][C][Y][X] = 1
			
			#maskを生成
			mask_tensor = torch.zeros(len(labels), 10, len(labels[0]), len(labels[0][0]))
			mask_tensor4net = torch.zeros(len(labels), 1, len(labels[0]), len(labels[0][0]))
			
			for B in range(len(labels)):
				for C in range(10):
					for Y in range(len(labels[B])):
						for X in range(len(labels[B][Y])):
							if mask[Y][X] != 0:
								mask_tensor[B][C][Y][X] = 1 
								mask_tensor4net[B][0][Y][X] = 1
		
			"""
				for B in range(len(labels)):
					for C in range(10):
						for Y in range(len(labels[B])):
							for X in range(len(labels[B][Y])):
								if labels[B][Y][X] != 0:
									mask_tensor[B][C][Y][X] = 1
			"""
			if args.gpu >= 0:
				inputs = inputs.to(device)
				labels = labels.to(device)
				targets = targets.to(device)
				mask_tensor = mask_tensor.to(device)
				mask_tensor4net = mask_tensor4net.to(device)
			# Reset the parameter gradients
			optimizer.zero_grad()

			#print("before forward")
			# Forward
			inputs.requires_grad = True
			outputs = net(inputs, mask_tensor4net)
			#print("outputs.requires_grad:", outputs.requires_grad)

			# Predict the label
			print("")
			#print("outputs:", outputs.size())
			#print("targets:", targets.size())
			#print("labels:", labels.size())
			#print("mask_tensor:", mask_tensor.size())
			_, predicted = torch.max(outputs, 1)

			"""
				print("predicted:",predicted.size())
				print("labels:",labels.size())
				print("outputs:",outputs.size())
			"""
				#maskを掛ける
				#outputs = outputs * mask_tensor
				#targets = targets * mask_tensor
			"""
				for E in range(len(outputs)):
					for Y in range(len(outputs[E][0])):
						for X in range(len(outputs[E][0][Y])):
							if mask[Y][X] == 0:
								outputs[E][0][Y][X] = 0.1
								for cl in range(1, 10):
									outputs[E][cl][Y][X] = -0.1
				
				for E in range(len(predicted)):
					for Y in range(len(predicted[E])):
						for X in range(len(predicted[E][Y])):
							if mask[Y][X] == 0:
								labels[E][Y][X] = predicted[E][Y][X]
			"""
			# Check whether estimation is right
			c = (predicted == labels).squeeze() ##この辺怪しい
			#print("c:", c.size())
			
			for i in range(len(predicted)):
				for j in range(len(predicted[i])):
					for k in range(len(predicted[i][j])):
						if len(predicted) == 1:
							correct_train += c[j][k].item()
							total_train += 1
						else:
							correct_train += c[i][j][k].item()
							total_train += 1
			# Backward + Optimize
			#loss = criterion(outputs, targets)
			loss = criterion(outputs, targets, mask_tensor)
			print(loss)
			loss.backward()
			optimizer.step()
			# Add loss
			running_loss += loss.item()
			print("trainloader_",s,"loss:",loss.item())

		# Report loss of the epoch
		print('[epoch %d] loss: %.3f' % (ep + 1, running_loss))

		# Save the model
		if (ep + 1) % args.frequency == 0:
			path = args.out + "/model_" + str(ep + 1)
			torch.save(net.state_dict(), path)

		# Validation
		"""
			with torch.no_grad():
				for data in valloader:
					images, labels = data
					if args.gpu >= 0:
						images = images.to(device)
						labels = labels.to(device)
					outputs = net(images, mask_tensor4net)
					# Predict the label
					_, predicted = torch.max(outputs, 1)
					# Check whether estimation is right
					
					c = (predicted == labels).squeeze()
					#print("c:", c.size())
					#print("predicted:", predicted.size())
					for i in range(len(predicted)):
						for j in range(len(predicted[i])):
							for k in range(len(predicted[i][j])):
								if len(predicted) == 1:
									correct_val += c[j][k].item()
									total_val += 1
								else:
									correct_val += c[i][j][k].item()
									total_val += 1
		"""

		# Record result
		"""
			x.append(ep + 1)
			ac_train.append(100 * correct_train / total_train)
			ac_val.append(100 * correct_val / total_val)
		"""

	print('Finished Training')
	path = args.out + "/model_final"
	torch.save(net.state_dict(), path)

	# Draw graph
	"""
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(x, ac_train, label='Training')
	ax.plot(x, ac_val, label='Validation')
	ax.legend()
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy [%]")
	ax.set_ylim(0, 100)

	plt.savefig(args.out + '/accuracy_cifar.png')
	#plt.show()
	"""

if __name__ == '__main__':
	main()