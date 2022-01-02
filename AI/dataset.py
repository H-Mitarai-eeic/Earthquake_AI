import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import math 

class MyDataSet(Dataset):
	def __init__(self, channels=2, root=None, train=True, transform=None, ID=None, mesh_size=(64, 64), depth_max=1000, expand=10):
		self.root = root
		self.transform = transform
		self.channels = channels
		self.mesh_size = mesh_size	#tuple x, y
		self.depth_max = depth_max	#in km
		self.expand = expand
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

		for Y in range(y - self.expand, y + self.expand + 1):
			for X in range(x - self.expand, x + self.expand + 1):
				if X >= 0 and X < self.mesh_size[0] and Y >= 0 and Y < self.mesh_size[1]:
					img[0][Y][X] = mag / 9
					img[1][Y][X] = depth / self.depth_max
		return img, lbl_data
