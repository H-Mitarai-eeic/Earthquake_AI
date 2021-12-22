
import argparse
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from network import CifarCNN

from dataset import MyDataSet

from MyGanNet import MYFCN4gan
from MyGanNet import MyDiscriminator
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
	parser.add_argument('--dataset', '-d', default='data1000_honshu6464_mag50/',
						help='Root directory of dataset')
	parser.add_argument('--mask', '-mask', default='ObservationPointsMap_honshu6464.csv',
						help='Root directory of dataset')
	args = parser.parse_args()
	print("train_eq_gan")
	print("output: " ,args.out)
	print("dataset: ", args.dataset)
	print("mask: ", args.mask)
	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('')
	
	data_channels = 1
	lr = 0.001
	#noise_div = 10
	mesh_size = (64, 64, 10)
	depth_max = 800
	print("mesh_size: ", mesh_size)
	print("data_channels", data_channels)
	print("depth_max", depth_max)
	print("learning rate: ", lr)
	print('')
	# Set up a neural network to train
	net = MYFCN4gan(in_channels=data_channels + 0, out_channels=1, mesh_size=mesh_size)
	D = MyDiscriminator(in_channels = 0 + data_channels, mesh_size=mesh_size)

	# Load designated network weight
	if args.resume:
		net.load_state_dict(torch.load(args.resume))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)
		D = D.to(device)

	# Setup a loss and an optimizer
	Loss = nn.BCELoss()
	net_optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
	D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
	trainvalset = MyDataSet(channels=data_channels, root=args.dataset, train=True, transform=transform, mesh_size=mesh_size)
	
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
	
	# Train
	for ep in range(args.epoch):  # Loop over the dataset multiple times

		running_loss = 0.0
		correct_train = 0
		#total_train = 0
		correct_val = 0
		total_val = 0
		running_D_loss = 0
		running_net_loss = 0

		for s, data in enumerate(trainloader, 0):
			# Get the inputs; data is a list of [inputs, labels]
			#Dの学習
			epic_data, real_data = data
			#noise = (torch.rand(real_data.shape[0], 1, real_data.shape[2], real_data.shape[3]) - 0.5) / 0.5 / noise_div

			if args.gpu >= 0:
				real_data = real_data.to(device)
				epic_data = epic_data.to(device)
				#noise = noise.to(device)

			real_data_epic_data = torch.cat((real_data, epic_data), dim = 2)
			if args.gpu >= 0:
				real_data_epic_data = real_data_epic_data.to(device)
			real_outputs = D(real_data_epic_data)	#本物をDで評価　epicenter_data追加

			real_label = torch.ones(real_data.shape[0], 1)	# 正解レベル all 1
			#real_label = torch.rand(real_data.shape[0], 1) * 0.55 + 0.675 # 正解レベル 0.675 ~ 1.225
			"""
			epic_data_noise = torch.cat((epic_data, noise), dim = 2)
			if args.gpu >= 0:
				epic_data_noise = epic_data_noise.to(device)
			"""
			#predicted_data = net(epic_data_noise)	#偽物生成　ノイズ追加
			predicted_data = net(epic_data)

			predicted_data_epic_data = torch.cat((predicted_data, epic_data), dim = 2)
			if args.gpu >= 0:
				predicted_data_epic_data = predicted_data_epic_data.to(device)
			predicted_data_outputs = D(predicted_data_epic_data)	#偽物をDで評価epicenter_data追加

			predicted_data_label = torch.zeros(predicted_data.shape[0], 1)
			#predicted_data_label = torch.rand(predicted_data.shape[0], 1) * 0.35

			if args.gpu >= 0:
				real_label = real_label.to(device)
				predicted_data_label = predicted_data_label.to(device)

			outputs = torch.cat((real_outputs.squeeze(), predicted_data_outputs.squeeze()), 0)
			targets = torch.cat((real_label.squeeze(), predicted_data_label.squeeze()), 0)

			D_loss = Loss(outputs, targets)	#Dの学習
			D_optimizer.zero_grad()
			D_loss.backward()
			D_optimizer.step()

			#networkの学習
			#noise = (torch.rand(real_data.shape[0], 1, real_data.shape[2], real_data.shape[3]) - 0.5) / 0.5 / noise_div
			#if args.gpu >= 0:
				#noise = noise.to(device)
			"""
			epic_data_noise = torch.cat((epic_data, noise), dim = 2)
			if args.gpu >= 0:
				epic_data_noise = epic_data_noise.to(device)
			"""
			#predicted_data = net(epic_data_noise)	#偽物生成　ノイズ追加
			predicted_data = net(epic_data)
			
			predicted_data_epic_data = torch.cat((predicted_data, epic_data), dim = 2)
			if args.gpu >= 0:
				predicted_data_epic_data = predicted_data_epic_data.to(device)

			predicted_data_outputs = D(predicted_data_epic_data)	#偽物をDで評価epicenter_data追加
			predicted_data_targets = torch.ones(predicted_data.shape[0], 1)

			if args.gpu >= 0:
				real_label = real_label.to(device)
				predicted_data_targets = predicted_data_targets.to(device)

			net_loss = Loss(predicted_data_outputs.squeeze(), predicted_data_targets.squeeze())
			net_optimizer.zero_grad()	#ネットワークの学習
			net_loss.backward()
			net_optimizer.step()

			# Add loss
			running_D_loss += D_loss.item()
			running_net_loss += net_loss.item()
			print("trainloader_",s,"D_loss:",D_loss.item(), "net_loss:", net_loss.item())

		# Report loss of the epoch
		print('[epoch %d] D_loss: %.3f, net_loss: %.3f' % (ep + 1, running_D_loss, running_net_loss))
		print('')

		# Save the model
		if (ep + 1) % args.frequency == 0:
			path = args.out + "/model_" + str(ep + 1)
			torch.save(net.state_dict(), path)
				

	print('Finished Training')
	path = args.out + "/model_final"
	torch.save(net.state_dict(), path)

if __name__ == '__main__':
	main()