
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

# from network import CifarCNN
from Linear import Linear
from dataset import MyDataSet


def main():
	parser = argparse.ArgumentParser(description='Earthquaker')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=20,
						help='Number of sweeps over the training data')
	parser.add_argument('--frequency', '-f', type=int, default=-1,
						help='Frequency of taking a snapshot')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='./result',
						help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	parser.add_argument('--dataset', '-d', default='../data',
						help='Root directory of dataset')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')

	# Set up a neural network to train
	net = Linear(10)
	# Load designated network weight
	if args.resume:
		net.load_state_dict(torch.load(args.resume))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		print("GPU using")
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)

	# Setup a loss and an optimizer
	criterion = nn.CrossEntropyLoss()
	# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9) #lr 0.001
	lr = 1
	optimizer = optim.Adam(net.parameters(), lr)

	# Load the data

	transform = transforms.Compose([transforms.ToTensor()])

	trainvalset = MyDataSet(root=args.dataset, train=True, transform=transform)
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

			if args.gpu >= 0:
				inputs = inputs.to(device)
				labels = labels.to(device)
			# Reset the parameter gradients
			optimizer.zero_grad()

			# Forward
			outputs = net(inputs)
	
			# Predict the label
			_, predicted = torch.max(outputs, 1)
			# Check whether estimation is right
			c = (predicted == labels).squeeze() ##この辺怪しい

			for i in range(len(predicted)):
				for j in range(len(predicted[i])):
					for k in range(len(predicted[i][j])):
						correct_train += c[i][j][k].item()
						total_train += 1
			# Backward + Optimize
			loss = criterion(outputs, labels)
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
		with torch.no_grad():
			for data in valloader:
				images, labels = data
				if args.gpu >= 0:
					images = images.to(device)
					labels = labels.to(device)
				outputs = net(images)
				# Predict the label
				_, predicted = torch.max(outputs, 1)
				# Check whether estimation is right
				c = (predicted == labels).squeeze()
				for i in range(len(predicted)):
					for j in range(len(predicted[i])):
						for k in range(len(predicted[i][j])):
							correct_val += c[i][j][k].item()
							total_val += 1

		# Record result
		x.append(ep + 1)
		ac_train.append(100 * correct_train / total_train)
		ac_val.append(100 * correct_val / total_val)

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
	ax.set_ylim(0, 100)

	plt.savefig(args.out + '/accuracy_earthquaker.png')
	#plt.show()

if __name__ == '__main__':
	main()
