import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math 

class MyDataSet(Dataset):
	def __init__(self, channels=2, root=None, train=True, transform=None, ID=None, mesh_size=(64, 64), depth_max_reg=600, depth_max_cls=1000, mag_degree=1, depth_degree=1, cross_degree=0, dim_cls=2):
		self.root = root
		self.transform = transform
		self.channels = channels
		self.mesh_size = mesh_size	#tuple x, y
		self.depth_max_reg = depth_max_reg	#in km
		self.depth_max_cls = depth_max_cls	#in km
		self.mag_degree = mag_degree
		self.depth_degree = depth_degree
		self.cross_degree = cross_degree

		self.input_width = 21
		self.dim = dim_cls

		mode = "train" if train else "test"
		
		#全てのデータのパスを入れる
		data_dir = os.path.join(self.root, mode)
		if mode == "train":
			self.all_data = glob.glob(data_dir + "/*")
		elif mode == "test" and ID != None:
			self.all_data = glob.glob(data_dir + "/" + ID + ".csv")
		elif mode == "test" and ID == None:
			self.all_data = glob.glob(data_dir + "/*")
		#all_dataは一次元配列

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, idx):
		with open(self.all_data[idx], "r") as f:
			txt = f.readlines()[0]
		x, y, depth, mag = txt.split(",")
		x, y, depth, mag = int(x), int(y), float(depth), float(mag)
		lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype=float, skiprows=1)

		#for regression
		img_reg = torch.zeros(self.channels, self.mesh_size[1], self.mesh_size[0])

		i = 0
		img_reg[i][y][x] = 1
		i += 1

		for j in range(1, self.mag_degree + 1):
			img_reg[i][y][x] = (mag / 9)**j
			i += 1
		for j in range(1, self.depth_degree + 1):
			img_reg[i][y][x] = (depth / self.depth_max_reg)**j
			i += 1
		for j in range(1, self.cross_degree // 2 + 1):
			img_reg[i][y][x] = ((mag / 9) * (depth / self.depth_max_reg)) ** (2*j)
			i += 1

		#for classification
		len_data = len(lbl_data)
		#img_cls = torch.zeros(2, self.mesh_size[1], self.mesh_size[0])
		img_cls = torch.zeros(self.dim, self.mesh_size[1], self.mesh_size[0])
		half = self.input_width // 2
		for i in range(x - half, x + half + 1):
			for j in range(y - half, y + half + 1):
				if 0 <= i < len_data and 0 <= j < len_data:
					img_cls[0][i][j] = depth / self.depth_max_cls
					#img_cls[1][i][j] = (mag / 10) ** 9
					for k in range(self.dim - 1):
						img_cls[k + 1][i][j] = (mag / 10) ** (k+1)

		return img_reg, img_cls, lbl_data
