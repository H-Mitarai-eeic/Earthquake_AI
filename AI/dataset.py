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
		#print(lbl_data)
		img = torch.zeros(2, len(lbl_data), len(lbl_data))
		img[0][x][y] = depth
		img[1][x][y] = mag
		return img, lbl_data



