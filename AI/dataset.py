import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math 

class MyDataSet(Dataset):
	def __init__(self, channels=2, root=None, train=True, transform=None, ID=None, mesh_size=(64, 64), depth_max=1000, mag_degree=1, depth_degree=1, cross_degree=0):
		self.root = root
		self.transform = transform
		self.channels = channels
		self.mesh_size = mesh_size	#tuple x, y
		self.depth_max = depth_max	#in km
		self.mag_degree = mag_degree
		self.depth_degree = depth_degree
		self.cross_degree = cross_degree

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

		img = torch.zeros(self.channels, self.mesh_size[1], self.mesh_size[0])

		i = 0
		img[i][y][x] = 1
		i += 1

		for j in range(1, self.mag_degree + 1):
			img[i][y][x] = (mag / 9)**j
			i += 1
		for j in range(1, self.depth_degree + 1):
			img[i][y][x] = (depth / self.depth_max)**j
			i += 1
		for j in range(1, self.cross_degree // 2 + 1):
			img[i][y][x] = ((mag / 9) * (depth / self.depth_max)) ** (2*j)
			i += 1

		return img, lbl_data
