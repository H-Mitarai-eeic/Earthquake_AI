import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np

class MyDataSet(Dataset):
	def __init__(self, channels=3, root=None, train=True, transform=None, ID=None, mesh_size=64):
		self.root = root
		self.transform = transform
		self.channels = channels
		self.mesh_size = mesh_size
		mode = "train" if train else "test"
		
		#全てのデータのパスを入れる
		data_dir = os.path.join(self.root, mode)
		if mode == "train":
			self.all_data = glob.glob(data_dir + "/*")
		elif mode == "test":
			self.all_data = glob.glob(data_dir + "/" + ID + ".csv")
		#all_dataは一次元配列

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, idx):
		with open(self.all_data[idx], "r") as f:
			txt = f.readlines()[0]
		x, y, depth, mag = txt.split(",")
		x, y, depth, mag = int(x), int(y), float(depth), float(mag)
		lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype=int, skiprows=1)
		img = torch.zeros(self.channels)
		img[0] = (x / self.mesh_size) * (10**mag)
		img[1] = (y / self.mesh_size) * (10**mag)
		img[2] = (depth / 1000) * (10**mag)
		return img, lbl_data

class MyDataSet4gan(Dataset):
	def __init__(self, channels=2, root=None, train=True, transform=None, ID=None):
		self.root = root
		self.transform = transform
		self.channels = channels
		mode = "train" if train else "test"
		
		#全てのデータのパスを入れる
		data_dir = os.path.join(self.root, mode)
		if mode == "train":
			self.all_data = glob.glob(data_dir + "/*")
		elif mode == "test":
			self.all_data = glob.glob(data_dir + "/" + ID + ".csv")
		#all_dataは一次元配列

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, idx):
		with open(self.all_data[idx], "r") as f:
			txt = f.readlines()[0]
		x, y, depth, mag = txt.split(",")
		x, y, depth, mag = int(x), int(y), float(depth), float(mag)
		lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype=int, skiprows=1)

		labels = torch.zeros(1, len(lbl_data), len(lbl_data[0]))
		for Y in range(len(lbl_data)):
			for X in range(len(lbl_data[0])):
				labels[0][Y][X] = int(lbl_data[Y][X].item())

		img = torch.zeros(self.channels, len(lbl_data), len(lbl_data))
		img[0][y][x] = depth / 1000
		img[1][y][x] = mag / 10

		return img, labels

