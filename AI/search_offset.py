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

from mycfc2D import MYFCN
from Linear import Linear

import csv
import copy

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Earthquaker')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--reg_pre_II_map_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--reg_pre_II_t_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--cls_pre_II_map_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--cls_pre_II_t_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--obs_II_map_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--obs_II_t_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--obs_SI_map_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--obs_SI_t_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--obs_SI_t_mask', '-d', default='results/model4osaka/test_data',
						help='data')
	parser.add_argument('--out', '-o', default='serch_offset_results',
						help='Root directory of outputfile')					
	parser.add_argument('--mask', '-mask', default='ObservationPointsMap_honshu6464.csv',
						help='Root directory of dataset')
	
	parser.add_argument('--use_traindata', '-u', default=1, type=int,
						help='Root directory of dataset')

	parser.add_argument('--start', '-s', default=1, type=int,
						help='Root directory of dataset')
	parser.add_argument('--step', '-p', default=1, type=int,
						help='Root directory of dataset')
	parser.add_argument('--end', '-e', default=1, type=int,
						help='Root directory of dataset')
	args = parser.parse_args()

	#open mask
	with open(args.mask, "r") as f_mask:
		reader = csv.reader(f_mask)
		mask = [[int(row2) for row2 in row] for row in reader]

	if args.use_traindata == 0:
		train = False
		print("use test data")
	else:
		train = True
		print("use train data")
	
	
	# Test
	print("Test")
	correct = 0
	total = 0

	counter = 1

	targets_map_mask_SI_list = [] #SI means 震度階級
	predict_map_mask_SI_list = []

	targets_t_mask_SI_list = []
	predict_t_mask_SI_list = []

	targets_no_mask_SI_list = []
	predict_no_mask_SI_list = []

	targets_map_mask_II_list = [] #II means 計測震度
	predict_map_mask_II_list = []

	targets_t_mask_II_list = []
	predict_t_mask_II_list = []

	targets_no_mask_II_list = []
	predict_no_mask_II_list = []

	targets_map_mask_cls_list = [] #II means 計測震度
	predict_map_mask_cls_list = []

	targets_t_mask_cls_list = []
	predict_t_mask_cls_list = []

	targets_no_mask_cls_list = []
	predict_no_mask_cls_list = []

	with torch.no_grad():
		for data in testloader:
			print("test #", counter)
			# Get the inputs; data is a list of [inputs, labels]
			images4reg, images4cls, labels = data
			if args.gpu >= 0:
				images4reg = images4reg.to(device)
				images4cls = images4cls.to(device)
				labels = labels.to(device)
				
			# Forward
			outputs_reg = net_reg(images4reg)
			outputs_cls_likelihood = net_cls(images4cls)
			_, outputs_cls = torch.max(outputs_cls_likelihood, 1)
			
			predicted_SI = [[[0 for i in range(len(labels[0][0]))] for j in range(len(labels[0]))] for k in range(len(labels))]
			predicted_II = [[[0 for i in range(len(labels[0][0]))] for j in range(len(labels[0]))] for k in range(len(labels))]
			cls_label = [[[0 for i in range(len(labels[0][0]))] for j in range(len(labels[0]))] for k in range(len(labels))]

			
			
			for B in range(len(outputs_reg)):
				for Y in range(len(outputs_reg[B])):
					for X in range(len(outputs_reg[B][Y])):
						if labels[B][Y][X].item() > 0.1:
							cls_label[B][Y][X] = 1

						predicted_SI[B][Y][X] = InstrumentalIntensity2SesimicIntensity(outputs_reg[B][Y][X].item())
						predicted_II[B][Y][X] = outputs_reg[B][Y][X].item()
						
						targets_no_mask_II_list.append(labels[B][Y][X].item())
						predict_no_mask_II_list.append(predicted_II[B][Y][X])

						targets_no_mask_SI_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
						predict_no_mask_SI_list.append(predicted_SI[B][Y][X])

						targets_no_mask_cls_list.append(cls_label[B][Y][X])
						predict_no_mask_cls_list.append(outputs_cls[B][Y][X])
						
						if mask[Y][X] > 0:
							targets_map_mask_II_list.append(labels[B][Y][X].item())
							predict_map_mask_II_list.append(predicted_II[B][Y][X])

							targets_map_mask_SI_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
							predict_map_mask_SI_list.append(predicted_SI[B][Y][X])

							targets_map_mask_cls_list.append(cls_label[B][Y][X])
							predict_map_mask_cls_list.append(outputs_cls[B][Y][X])

						if labels[B][Y][X].item() > 0.1:
							targets_t_mask_II_list.append(labels[B][Y][X].item())
							predict_t_mask_II_list.append(predicted_II[B][Y][X])

							targets_t_mask_SI_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
							predict_t_mask_SI_list.append(predicted_SI[B][Y][X])

							targets_t_mask_cls_list.append(cls_label[B][Y][X])
							predict_t_mask_cls_list.append(outputs_cls[B][Y][X])
			
			counter += 1
			

	with open(args.out + '/Pre_II_map_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_map_mask_II_list)
	with open(args.out + '/Obs_II_map_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_map_mask_II_list)
	with open(args.out + '/Pre_II_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_t_mask_II_list)
	with open(args.out + '/Obs_II_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_t_mask_II_list)
	with open(args.out + '/Pre_II_no_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_no_mask_II_list)
	with open(args.out + '/Obs_II_no_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_no_mask_II_list)		
	
	with open(args.out + '/Pre_SI_map_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_map_mask_SI_list)
	with open(args.out + '/Obs_SI_map_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_map_mask_SI_list)
	with open(args.out + '/Pre_SI_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_t_mask_SI_list)
	with open(args.out + '/Obs_SI_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_t_mask_SI_list)
	with open(args.out + '/Pre_SI_no_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_no_mask_SI_list)
	with open(args.out + '/Obs_SI_no_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_no_mask_SI_list)	

	with open(args.out + '/Pre_cls_map_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_map_mask_cls_list)
	with open(args.out + '/Obs_cls_map_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_map_mask_cls_list)
	with open(args.out + '/Pre_cls_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_t_mask_cls_list)
	with open(args.out + '/Obs_cls_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_t_mask_cls_list)
	with open(args.out + '/Pre_cls_no_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_no_mask_cls_list)
	with open(args.out + '/Obs_cls_no_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_no_mask_cls_list)

	print("finished")	

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
	#自由度調整済み決定係数
    return 1-(1-r2_score(y_true, y_pred)) * (len(y_true)-1) / (len(y_true) - p - 1)

def ME(y_true, y_pred):
	#残差の平均
	me = 0
	n = len(y_true)
	for i in range(n):
		me += y_true[i] - y_pred[i]
	me = me / n
	return me

def RSS(y_true, y_pred):
	#残差平方和
	rss = 0
	for i in range(len(y_true)):
		rss += (y_true[i] - y_pred[i]) ** 2
	return rss

def MSE(y_true, y_pred):
	#平均二乗誤差
	rss = RSS(y_true, y_pred)
	n = len(y_true)
	mse = rss / n
	return mse

def RMSE(y_true, y_pred):
	#平均二乗偏差
	mse = MSE(y_true, y_pred)
	rmse = mse ** 0.5
	return rmse

def RSE(y_true, y_pred, p=2):
	#相対2乗誤差
	rss = RSS(y_true, y_pred)
	n = len(y_true)
	rse = (rss / (n-p-1))** 0.5
	return rse

def L1_LOSS(y_true, y_pred):
	#L1損失
	l1_loss = 0
	for i in range(len(y_true)):
		l1_loss += abs(y_true[i] - y_pred[i])
	return l1_loss

def MAE(y_true, y_pred):
	#平均絶対誤差
	l1_loss = L1_LOSS(y_true, y_pred)
	N = len(y_true)
	return l1_loss / N

if __name__ == '__main__':
	main()