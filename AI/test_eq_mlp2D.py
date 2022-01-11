import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import r2_score
import numpy as np

from dataset import MyDataSet
from mymlp2D import MYFCN
#from myloss import MyLoss

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
	parser.add_argument('--out', '-o', default='data100/',
						help='Root directory of outputfile')					
	parser.add_argument('--ID', '-i', default=None,
						help='Erthquake ID for input/output files')
	parser.add_argument('--mask', '-mask', default='ObservationPointsMap_honshu6464.csv',
						help='Root directory of dataset')
	parser.add_argument('-expand', '-expand', default=10,
						help='expantion')
	args = parser.parse_args()

	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print("model", args.model)
	print('')

	#open mask
	with open(args.mask, "r") as f_mask:
		reader = csv.reader(f_mask)
		mask = [[int(row2) for row2 in row] for row in reader]

	# Set up a neural network to test
	mesh_size = (64, 64)
	data_channels = 2
	depth_max = 600
	expand = int(args.expand)
	print("mesh_size: ", mesh_size)
	print("data_channels", data_channels)
	print("depth_max", depth_max)
	print("expand", expand)
	print("")
	net = MYFCN(in_channels=data_channels, mesh_size=mesh_size)
	# Load designated network weight
	print("loading Model...")
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net = net.to(device)
		net.load_state_dict(torch.load(args.model))
	else:
		net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])
	testset = MyDataSet(channels=data_channels, root=args.dataset, train=False, transform=transform, ID=args.ID, mesh_size=mesh_size, depth_max=depth_max, expand=expand)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

	# Test
	print("Test")
	correct = 0
	total = 0

	counter = 1
	#class_correct = list(0. for i in range(10))
	class_diff = list(0. for i in range(19))
	#class_total = list(0. for i in range(10))

	targets_list = []
	predict_list = []

	targets_masked_list = []
	predict_masked_list = []

	residuals_list= []
	targets_masked_InstrumentalIntensity_list = []
	predict_masked_InstrumentalIntensity_list = []

	with torch.no_grad():
		# 多分1つしかテストしないようになっているはず
		for data in testloader:
			#print("test #", counter)
			# Get the inputs; data is a list of [inputs, labels]
			images, labels = data
			if args.gpu >= 0:
				images = images.to(device)
				labels = labels.to(device)

			# Forward
			outputs = net(images)
			
			# Predict the label
			predicted = [[[0 for i in range(len(labels[0][0]))] for j in range(len(labels[0]))] for k in range(len(labels))]
			
			for B in range(len(outputs)):
				for Y in range(len(outputs[B])):
					for X in range(len(outputs[B][Y])):
#						if mask[Y][X] > 0:
						predicted[B][Y][X] = InstrumentalIntensity2SesimicIntensity(outputs[B][Y][X].item())
						targets_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
						predict_list.append(predicted[B][Y][X])
						if mask[Y][X] > 0:
							targets_masked_InstrumentalIntensity_list.append(labels[B][Y][X].item())
							predict_masked_InstrumentalIntensity_list.append(outputs[B][Y][X].item())
							residuals_list.append(labels[B][Y][X].item() - outputs[B][Y][X].item())

							targets_masked_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
							predict_masked_list.append(predicted[B][Y][X])

			
			for B in range(len(labels)):
				for Y in range(len(labels[B])):
					for X in range(len(labels[B][Y])):
						if mask[Y][X] != 0:
							label = InstrumentalIntensity2SesimicIntensity(labels[B][Y][X])
							predic = predicted[B][Y][X]
							class_diff_index = int(predic-label) + 9
							class_diff[class_diff_index] += 1
							total += 1 
			counter += 1
			

	# List of classes
	classes = ("0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7")
	classes_diff_ver = ("-9", "-8","-7","-6","-5","-4","-3","-2","-1","0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
	
	# Show accuracy
	print("予測震度と実際の震度のずれの分布")
	for i in range(19):
		print('%5s 階級 : %2d %% (total %d)' % (classes_diff_ver[i], 100 * class_diff[i] / total, class_diff[i]))

	#matthews corrcoef
	print("matthews corrcoef(マスクなし)", matthews_corrcoef(np.array(targets_list), np.array(predict_list)))
	print("matthews corrcoef(マスクあり)", matthews_corrcoef(targets_masked_list, predict_masked_list))
	#決定係数
	print("決定係数", r2_score(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list))
	#自由度調整済み決定係数
	print("自由度調整済み決定係数", adj_r2_score(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list, data_channels))
	#ピアソン相関係数
	print("ピアソン相関係数", np.corrcoef(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)[0, 1])
	#RSS
	print("RSS", RSS(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list))
	#RSE
	print("RSE", RSE(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list, data_channels))
	#L1 LOSS
	print("L1 LOSS", L1_LOSS(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list))
	#MAE
	print("MAE", MAE(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list))
	#residual plot
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(predict_masked_InstrumentalIntensity_list, residuals_list)
	ax.set_xlabel("Predicted Instrumental Intensities")
	ax.set_ylabel("Residuals")
	ax.set_ylim(min(residuals_list), max(residuals_list))
	ax.set_xlim(min(predict_masked_InstrumentalIntensity_list), max(predict_masked_InstrumentalIntensity_list))

	plt.savefig(args.out + '/ResidualPlot_polyCFC.png')

	#真値-予測値
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(predict_masked_InstrumentalIntensity_list, targets_masked_InstrumentalIntensity_list)
	ax.set_xlabel("Predicted Values")
	ax.set_ylabel("True Values")
	ax.set_ylim(min(targets_masked_InstrumentalIntensity_list), max(targets_masked_InstrumentalIntensity_list))
	ax.set_xlim(min(predict_masked_InstrumentalIntensity_list), max(predict_masked_InstrumentalIntensity_list))

	plt.savefig(args.out + '/Target-PredictedPlot_mlp.png')
	"""
	#csv出力
	predicted_map = copy.deepcopy(predicted)

	#print(predicted_map)
	with open(args.output + 'predicted/' + args.ID + '_predicted.csv', "w") as fo:
		writer = csv.writer(fo)
		writer.writerows(predicted_map)
	"""

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
		return 9

def adj_r2_score(y_true, y_pred, p=2):
    return 1-(1-r2_score(y_true, y_pred)) * (len(y_true)-1) / (len(y_true) - p - 1)

def RSS(y_true, y_pred):
	rss = 0
	for i in range(len(y_true)):
		rss += (y_true[i] - y_pred[i]) ** 2
	return rss

def RSE(y_true, y_pred, p=2):
	rss = RSS(y_true, y_pred)
	n = len(y_true)
	rse = (rss / (n-p-1))** 0.5
	return rse

def L1_LOSS(y_true, y_pred):
	l1_loss = 0
	for i in range(len(y_true)):
		l1_loss += abs(y_true[i] - y_pred[i])
	return l1_loss

def MAE(y_true, y_pred):
	l1_loss = L1_LOSS(y_true, y_pred)
	N = len(y_true)
	return l1_loss / N

if __name__ == '__main__':
	main()
