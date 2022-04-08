import argparse
import torch

from dataset import MyDataSet
from mycfc2D import MYFCN

import csv
import copy
import math

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: CIFAR-10')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--model', '-m', default='result_cfc4/model_100',
						help='Path to the model for test')
	parser.add_argument('--output', '-o', default='data100/',
						help='Root directory of outputfile')					
	parser.add_argument('--x', '-x', default=52,
						help='Erthquake ID for input/output files')
	parser.add_argument('--y', '-y', default=32,
						help='Erthquake ID for input/output files')
	parser.add_argument('--depth', '-d', default=10,
						help='Erthquake ID for input/output files')
	parser.add_argument('--mag', '-mag', default=9.0,
						help='Erthquake ID for input/output files')
	parser.add_argument('--kernel_size', '-kernel_size', default=125,
						help='Root directory of dataset')
	parser.add_argument('--mag_degree', '-mag_d', default=14,
						help='Root directory of dataset')
	parser.add_argument('--depth_degree', '-depth_d', default=14,
						help='Root directory of dataset')
	parser.add_argument('--cross_degree', '-cross_d', default=0,
						help='Root directory of dataset')
	args = parser.parse_args()

	# Set up a neural network to test
	mesh_size = (64, 64)
	mag_degree = int(args.mag_degree)
	depth_degree = int(args.depth_degree)
	cross_degree = int(args.cross_degree)
	data_channels = mag_degree + depth_degree + (cross_degree // 2) + 1
	depth_max = 600
	kernel_size = int(args.kernel_size)
	net = MYFCN(in_channels=data_channels, mesh_size=mesh_size, kernel_size=kernel_size)
	# Load designated network weight
	print("loading Model...")
	net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

	# Test
	print("test")
	# Get the inputs; data is a list of [inputs, labels]
	x = int(args.x)
	y = int(args.y)
	depth = float(args.depth)
	mag = float(args.mag)

	epicenter = torch.zeros(1, data_channels, mesh_size[1], mesh_size[0])

	i = 0
	epicenter[0][i][y][x] = 1
	i += 1

	for j in range(1, mag_degree + 1):
		epicenter[0][i][y][x] = (mag / 9)**j
		i += 1
	for j in range(1, depth_degree + 1):
		epicenter[0][i][y][x] = (depth / depth_max)**j
		i += 1
	for j in range(1, cross_degree // 2 + 1):
		epicenter[0][i][y][x] = ((mag / 9) * (depth / depth_max)) ** (2*j)
		i += 1

	# Forward
	outputs = net(epicenter)
	# Predict the label
	predicted = [[0 for i in range(mesh_size[1])] for j in range(mesh_size[0])]
	
	for Y in range(mesh_size[1]):
		for X in range(mesh_size[0]):
			predicted[Y][X] = InstrumentalIntensity2SesimicIntensity(outputs[0][Y][X].item())

	#csv出力
	with open('predict_resdults/predicted_data.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerows(predicted)
	with open('predict_results/predicted_data_.csv', "w") as fo:
		writer = csv.writer(fo)
		writer.writerows(predicted)

def InstrumentalIntensity2SesimicIntensity(II):
	if II < 0.5:
		return 0
	elif II < 1.5:
		return 1
	elif II < 2.5:
		return 2
	elif II < 3.5:
		return 3
	elif II < 4.5:
		return 4
	elif II < 5.0:
		return 5	#5-
	elif II < 5.5:
		return 6	#5+
	elif II < 6.0:
		return 7	#6-
	elif II < 6.5:
		return 8	#6+
	else:
		return 9	#7

def depth2Z(depth, depth_max, mesh_size):
	Z = int(mesh_size[2] * math.log(1 + depth, depth_max))
	if Z >= mesh_size[2]:
		return mesh_size[2] - 1
	else:
		return  Z
		
if __name__ == '__main__':
	main()
