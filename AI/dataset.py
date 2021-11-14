import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np

class MyCifarDataset(Dataset):
	def __init__(self, root=None, train=True, transform=None):
		self.root = root
		self.transform = transform
		mode = 'train' if train else 'test'
		# TODO class辞書の定義
		classes = {
			"airplane":0,
			"automobile":1,
			"bird":2,
			"cat":3,
			"deer":4,
			"dog":5,
			"frog":6,
			"horse":7,
			"ship":8,
			"truck":9
		}
		# TODO 画像パスとそのラベルのセットをself.all_dataに入れる
		self.all_data = []
		for cls in classes:
			cls_dir = os.path.join(self.root, mode, cls)

			img_path_list = glob.glob(cls_dir + "/*") #glob,os,pathlibなどのモジュールを使うことが考えられる。その際、cls_dirを用いるとよい。
			cls_label = classes[cls]
			cls_data = [[img_path, cls_label] for img_path in img_path_list]

			self.all_data.extend(cls_data) #all_dataは二次元配列

	def __len__(self):
		#データセットの数を返す関数
		return len(self.all_data)
		
	def __getitem__(self, idx):
		# TODO 画像とラベルの読み取り
		#self.all_dataを用いる
		img = Image.open(self.all_data[idx][0]).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		label = self.all_data[idx][1]
		return [img, label]


class MyDataSet(Dataset):
	def __init__(self, root=None, train=True, transform=None, ID=None):
		self.root = root
		self.transform = transform
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
		img = torch.zeros(2, len(lbl_data), len(lbl_data))
		"""
		img[0][x][y] = depth
		img[1][x][y] = mag
		"""
		for i in range(x):
			for j in range(y):
				img[0][i][j] = depth
				img[1][i][j] = mag / ((x-i)**2) / ((y-j)**2)
		return img, lbl_data



