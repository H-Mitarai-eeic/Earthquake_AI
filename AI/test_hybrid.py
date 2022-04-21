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
#from myloss import MyLoss

import csv
import copy

def main():
	parser = argparse.ArgumentParser(description='Pytorch example: Earthquaker')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--model_reg', '-mr', default='results/model4osaka/model_final_reg',
						help='Path to the model for test')
	parser.add_argument('--model_cls', '-mc', default='results/model4osaka/model_final_cls',
						help='Path to the model for test')
	parser.add_argument('--dataset', '-d', default='data_for_hokkaido_regression/',
						help='Root directory of dataset')
	parser.add_argument('--out', '-o', default='results/model4osaka/',
						help='Root directory of outputfile')					
	parser.add_argument('--ID', '-i', default=None,
						help='Erthquake ID for input/output files')
	parser.add_argument('--mask', '-mask', default='ObservationPointsMap_honshu6464.csv',
						help='Root directory of dataset')
	parser.add_argument('--kernel_size', '-kernel_size', default=125,
						help='Root directory of dataset')
	parser.add_argument('--mag_degree', '-mag_d', default=1,
						help='Root directory of dataset')
	parser.add_argument('--depth_degree', '-depth_d', default=1,
						help='Root directory of dataset')
	parser.add_argument('--cross_degree', '-cross_d', default=0,
						help='Root directory of dataset')
	parser.add_argument('--merge_offset', '-mo', type=float, default=-1,
						help='Root directory of dataset')
	args = parser.parse_args()

	#print('GPU: {}'.format(args.gpu))
	#print('# Minibatch-size: {}'.format(args.batchsize))
	#print("model", args.model)
	#print('')

	#open mask
	with open(args.mask, "r") as f_mask:
		reader = csv.reader(f_mask)
		mask = [[int(row2) for row2 in row] for row in reader]

	# Set up a neural network to test
	mesh_size = (64, 64)
	mag_degree = int(args.mag_degree)
	depth_degree = int(args.depth_degree)
	cross_degree = int(args.cross_degree)
	merge_offset = float(args.merge_offset)

	data_channels = mag_degree + depth_degree + (cross_degree // 2) + 1
	depth_max_reg = 600
	depth_max_cls = 1000
	dim_cls = 9
	kernel_size = int(args.kernel_size)
	net_reg = MYFCN(in_channels=data_channels, mesh_size=mesh_size, kernel_size=kernel_size)
	net_cls = Linear(n_class=2, dim=dim_cls)
	# Load designated network weight
	#print("loading Model...")
	# Set model to GPU
	if args.gpu >= 0:
		# Make a specified GPU current
		device = 'cuda:' + str(args.gpu)
		net_reg = net_reg.to(device)
		net_reg.load_state_dict(torch.load(args.model_reg))
		net_cls = net_cls.to(device)
		net_cls.load_state_dict(torch.load(args.model_cls))
	else:
		net_reg.load_state_dict(torch.load(args.model_reg, map_location=torch.device('cpu')))
		net_cls.load_state_dict(torch.load(args.model_cls, map_location=torch.device('cpu')))

	# Load the CIFAR-10
	transform = transforms.Compose([transforms.ToTensor()])
	testset = MyDataSet(channels=data_channels, root=args.dataset, train=False, transform=transform, ID=args.ID, mesh_size=mesh_size, depth_max_reg=depth_max_reg, depth_max_cls=depth_max_cls, mag_degree=mag_degree, depth_degree=depth_degree, cross_degree=cross_degree, dim_cls=dim_cls)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)
	
	
	# Test
	#print("Test")
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

	targets_t_masked_list = []
	predict_t_masked_list = []

	residuals_masked_list= []
	targets_masked_InstrumentalIntensity_list = []
	predict_masked_InstrumentalIntensity_list = []

	residuals_t_masked_list= []
	targets_t_masked_InstrumentalIntensity_list = []
	predict_t_masked_InstrumentalIntensity_list = []

	with torch.no_grad():
		for data in testloader:
			#print("test #", counter)
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
			
			# Predict the label
			predicted_SI = [[[0 for i in range(len(labels[0][0]))] for j in range(len(labels[0]))] for k in range(len(labels))]
			predicted_II = [[[0 for i in range(len(labels[0][0]))] for j in range(len(labels[0]))] for k in range(len(labels))]
			
			for B in range(len(outputs_reg)):
				for Y in range(len(outputs_reg[B])):
					for X in range(len(outputs_reg[B][Y])):
						if outputs_cls[B][Y][X].item() == 0:
							if merge_offset < 0:
								predicted_SI[B][Y][X] = 0
								predicted_II[B][Y][X] = 0
							else:
								predicted_SI[B][Y][X] = InstrumentalIntensity2SesimicIntensity(outputs_reg[B][Y][X].item() - merge_offset)
								predicted_II[B][Y][X] = outputs_reg[B][Y][X].item() - merge_offset
						else:
							predicted_SI[B][Y][X] = InstrumentalIntensity2SesimicIntensity(outputs_reg[B][Y][X].item())
							predicted_II[B][Y][X] = outputs_reg[B][Y][X].item()
						
						targets_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
						predict_list.append(predicted_SI[B][Y][X])
						
						if mask[Y][X] > 0:
							targets_masked_InstrumentalIntensity_list.append(labels[B][Y][X].item())
							predict_masked_InstrumentalIntensity_list.append(predicted_II[B][Y][X])
							residuals_masked_list.append(labels[B][Y][X].item() - predicted_II[B][Y][X])

							targets_masked_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
							predict_masked_list.append(predicted_SI[B][Y][X])
						if labels[B][Y][X].item() > 0.1:
							targets_t_masked_InstrumentalIntensity_list.append(labels[B][Y][X].item())
							predict_t_masked_InstrumentalIntensity_list.append(predicted_II[B][Y][X])
							residuals_t_masked_list.append(labels[B][Y][X].item() - predicted_II[B][Y][X])

							targets_t_masked_list.append(InstrumentalIntensity2SesimicIntensity(labels[B][Y][X].item()))
							predict_t_masked_list.append(predicted_SI[B][Y][X])
			
			for B in range(len(labels)):
				for Y in range(len(labels[B])):
					for X in range(len(labels[B][Y])):
						if mask[Y][X] != 0:
							label = InstrumentalIntensity2SesimicIntensity(labels[B][Y][X])
							predic = predicted_II[B][Y][X]
							class_diff_index = int(predic-label) + 9
							class_diff[class_diff_index] += 1
							total += 1 
			counter += 1
			

	# List of classes
	classes = ("0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7")
	classes_diff_ver = ("-9", "-8","-7","-6","-5","-4","-3","-2","-1","0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
	
	# Show accuracy
	"""
		print("予測震度と実際の震度のずれの分布")
		for i in range(19):
			print('%5s 階級 : %2d %% (total %d)' % (classes_diff_ver[i], 100 * class_diff[i] / total, class_diff[i]))
	"""
	#統計量
	mcc_without_mask = matthews_corrcoef(np.array(targets_list), np.array(predict_list))
	mcc_with_mask = matthews_corrcoef(targets_masked_list, predict_masked_list)
	mcc_with_t_mask = matthews_corrcoef(targets_t_masked_list, predict_t_masked_list)

	r2 = r2_score(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)
	adj_r2 = adj_r2_score(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list, data_channels)
	pcc = np.corrcoef(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)[0, 1]
	me = ME(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)
	rss = RSS(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)
	mse = MSE(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)
	rmse = RMSE(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)
	rse = RSE(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list, data_channels)
	l1_loss = L1_LOSS(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)
	mae = MAE(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)

	r2_t = r2_score(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)
	adj_r2_t = adj_r2_score(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list, data_channels)
	pcc_t = np.corrcoef(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)[0, 1]
	me_t = ME(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)
	rss_t = RSS(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)
	mse_t = MSE(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)
	rmse_t = RMSE(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)
	rse_t = RSE(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list, data_channels)
	l1_loss_t = L1_LOSS(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)
	mae_t = MAE(targets_t_masked_InstrumentalIntensity_list, predict_t_masked_InstrumentalIntensity_list)
	
	print(mag_degree, depth_degree, cross_degree, mcc_without_mask, mcc_with_mask, mcc_with_t_mask, r2, adj_r2, pcc, me, rss, mse, rmse, rse, l1_loss, mae, r2_t, adj_r2_t, pcc_t, me_t, rss_t, mse_t, rmse_t, rse_t, l1_loss_t, mae_t, sep=',')

	#matthews corrcoef
	print("matthews corrcoef(マスクなし)", mcc_without_mask)
	print("matthews corrcoef(マスクあり)", mcc_with_mask)
	#決定係数
	print("決定係数", r2)
	print("決定係数(t_mask)", r2_t)
	#自由度調整済み決定係数
	print("自由度調整済み決定係数", adj_r2)
	#ピアソン相関係数
	print("ピアソン相関係数", pcc)
	#ME (残差の平均)
	print("ME", me)
	#RSS(残差平方和)
	print("RSS", rss)
	#MSE(平均二乗誤差)
	print("MSE", mse)
	#RMSE(#平均二乗偏差)
	print("RMSE", rmse)
	#RSE（相対に乗誤差）
	print("RSE", rse)
	#L1 LOSS
	print("L1 LOSS", l1_loss)
	#MAE（平均絶対誤差）
	print("MAE", mae)

	
	#residual plot
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(predict_masked_InstrumentalIntensity_list, residuals_masked_list)
	ax.set_xlabel("Predicted Instrumental Intensities")
	ax.set_ylabel("Residuals")
	ax.set_ylim(min(residuals_masked_list), max(residuals_masked_list))
	ax.set_xlim(min(predict_masked_InstrumentalIntensity_list), max(predict_masked_InstrumentalIntensity_list))

	plt.savefig(args.out + '/ResidualPlot_polycfc2D.png')

	#予測値-真値
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(targets_masked_InstrumentalIntensity_list, predict_masked_InstrumentalIntensity_list)
	ax.set_ylabel("Predicted Values")
	ax.set_xlabel("True Values")
	ax.set_xlim(min(targets_masked_InstrumentalIntensity_list), max(targets_masked_InstrumentalIntensity_list))
	ax.set_ylim(min(predict_masked_InstrumentalIntensity_list), max(predict_masked_InstrumentalIntensity_list))

	plt.savefig(args.out + '/Pre-Obs_polycfc2D.png')
	
	#print(predicted_map)
	with open(args.out + '/Pre_II_with_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_masked_InstrumentalIntensity_list)
	with open(args.out + '/Obs_II_with_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_masked_InstrumentalIntensity_list)
	with open(args.out + '/Pre_II_with_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_t_masked_InstrumentalIntensity_list)
	with open(args.out + '/Obs_II_with_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_t_masked_InstrumentalIntensity_list)
	
	with open(args.out + '/Pre_Class_without_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_list)
	with open(args.out + '/Obs_Class_without_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_list)
	with open(args.out + '/Pre_Class_with_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_masked_list)
	with open(args.out + '/Obs_Class_with_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_masked_list)
	with open(args.out + '/Pre_Class_with_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_t_masked_list)
	with open(args.out + '/Obs_Class_with_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(targets_t_masked_list)

	with open(args.out + '/Residual_with_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_masked_InstrumentalIntensity_list)
	with open(args.out + '/Residual_with_t_mask.csv', "w") as fo:
		writer = csv.writer(fo, lineterminator=',')
		writer.writerow(predict_t_masked_InstrumentalIntensity_list)
	

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