import argparse
import torch

from dataset import MyDataSet
from myc5fc2D import MYFCN

import csv
import copy
import math

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: CIFAR-10')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--model', '-m', default='result_c5fc2D_1/model_100',
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

	args = parser.parse_args()

	# Set up a neural network to test
	mesh_size = (64, 64)
	data_channels = 2
	depth_max = 600
	net = MYFCN(in_channels=data_channels, mesh_size=mesh_size)
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
	epicenter[0][0][y][x] = mag / 9
	epicenter[0][1][y][x] = depth / depth_max

	# Forward
	outputs = net(epicenter)
	# Predict the label
	predicted = [[0 for i in range(mesh_size[1])] for j in range(mesh_size[0])]
	
	for Y in range(mesh_size[1]):
		for X in range(mesh_size[0]):
			predicted[Y][X] = InstrumentalIntensity2SesimicIntensity(outputs[0][Y][X].item())

	#csv出力
	with open('predicted_data.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerows(predicted)
	with open('predicted_data_.csv', "w") as fo:
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
