import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np



class MyDataSet(Dataset):
	def __init__(self, root=None, train=True, transform=None):
		self.root = root
		self.transform = transform
		mode = "train" if train else "test"
		
		#全てのデータのパスを入れる
		data_dir = os.path.join(self.root, mode)
		self.all_data = glob.glob(data_dir + "/*")
		#all_dataは一次元配列

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, idx):
		with open(self.all_data[idx], "r") as f:
			txt = f.readlines()[0]
		x, y, depth, mag = txt.split(",")
		x, y, depth, mag = int(x), int(y), float(depth), float(mag)
		lbl_data = np.loadtxt(self.all_data[idx], delimiter=',', dtype=int, skiprows=1)
		lbl_data = np.where(lbl_data > 5, 5, lbl_data)
		#print(lbl_data)
		len_data = len(lbl_data)
		img = torch.zeros(2, len(lbl_data), len(lbl_data))
		for i in range(x-10, x+11):
			for j in range(y-10, y+11):
				if 0 <= i < len_data and 0 <= j < len_data:
					img[0][i][j] = depth / 1000
					img[1][i][j] = mag / 10
		# img[0][x][y] = depth / 1000
		# img[1][x][y] = mag / 10
		return img, lbl_data



