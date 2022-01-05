import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np

input_width = 21  # 奇数にして


class MyDataSet(Dataset):
    def __init__(self, root=None, train=True, transform=None):
        self.root = root
        self.transform = transform
        mode = "train" if train else "test"

        # 全てのデータのパスを入れる
        data_dir = os.path.join(self.root, mode)
        self.all_data = glob.glob(data_dir + "/*")
        # all_dataは一次元配列

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        with open(self.all_data[idx], "r") as f:
            txt = f.readlines()[0]
        x, y, depth, mag = txt.split(",")
        x, y, depth, mag = int(x), int(y), float(depth), float(mag)
        lbl_data = np.loadtxt(
            self.all_data[idx], delimiter=',', dtype=int, skiprows=1)
        # print(lbl_data)
        len_data = len(lbl_data)
        img = torch.zeros(2, len(lbl_data), len(lbl_data))
        half = input_width//2
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                if 0 <= i < len_data and 0 <= j < len_data:
                    img[0][i][j] = depth / 1000
                    img[1][i][j] = mag**9 / 1000000000
        # img[0][x][y] = depth / 1000
        # img[1][x][y] = mag / 10
        return img, lbl_data
