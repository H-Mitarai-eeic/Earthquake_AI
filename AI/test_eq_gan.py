import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#from fcn8s import FCN8s
#from fcn32s import FCN32s
from dataset import MyDataSet4gan
from MyGanNet import MYFCN4gan

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

	data_channels = 2
	lr = 0.001
	noise_div = 100
	with_noise = 1
	# Set up a neural network to train
	net = MYFCN4gan(in_channels=data_channels + with_noise, n_class=1)
	# Load designated network weight
	net.load_state_dict(torch.load(args.model))
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
	testset = MyDataSet4gan(channels=data_channels, root=args.dataset, train=False, transform=transform, ID=args.ID)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
											 shuffle=False, num_workers=2)

	# Test
	correct = 0
	total = 0
	class_diff = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		# 多分1つしかテストしないようになっているはず
		for data in testloader:
			# Get the inputs; data is a list of [inputs, labels]
			epic_data, real_data = data
			noise = (torch.rand(real_data.shape[0], 1, real_data.shape[2], real_data.shape[3]) - 0.5) / 0.5 / noise_div
			epic_data_noise = torch.cat((epic_data, noise), dim = 1)
			if args.gpu >= 0:
				real_data = real_data.to(device)
				epic_data = epic_data.to(device)
				noise = noise.to(device)
				epic_data_noise = epic_data_noise.to(device)

			if with_noise == 1:
				predicted_data = net(epic_data_noise)	#偽物生成　ノイズ追加
			else:
				predicted_data = net(epic_data)
			predicted_data = predicted_data.squeeze()
			real_data = real_data.squeeze()
			print("realdata: ", real_data.size())
			print("predicted_data: ", predicted_data.size())
			for Y in range(len(real_data)):
				for X in range(len(real_data[Y])):
					real_data_tmp = real_data[Y][X].item()
					predic = round(predicted_data[Y][X].item())
					#print("predic: ", predic, "real: ", real_data_tmp)
					class_diff_index = int(abs(real_data_tmp - predic))
					if class_diff_index > 9:
						class_diff_index = 9
					elif class_diff_index < 0:
						class_diff_index = 0
					class_diff[class_diff_index] += 1
					#total += 1 
		

	# List of classes
	classes_diff_ver = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
	
	# Show accuracy
	print("予測震度と実際の震度のずれの分布")
	for i in range(10):
		print('%5s 階級 : %2d 箇所' % (classes_diff_ver[i], class_diff[i]))


	#csv出力
	predicted_map = predicted_data.clone().squeeze().tolist()
	for Y in range(len(predicted_map)):
		for X in range(len(predicted_map[Y])):
			predicted_map[Y][X] = round(predicted_map[Y][X])

	with open(args.output + 'predicted/' + args.ID + '_predicted.csv', "w") as fo:
		writer = csv.writer(fo)
		writer.writerows(predicted_map)
	"""
	for i in range(10):
		print("class", i ,"at 10 10:", outputs[0][i][10][10])
		print("class", i, "at 20 20:", outputs[0][i][20][20])
	"""
if __name__ == '__main__':
	main()
