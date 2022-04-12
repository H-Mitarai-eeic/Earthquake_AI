import os
from turtle import width
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from math import exp

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
# from network import CifarCNN
from sklearn.metrics import matthews_corrcoef

len_data = 64
score = 0

class Linear(nn.Module):
    def __init__(self, n_class=10, dim=2):
        super(Linear, self).__init__()
        # self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.linear = nn.Linear(dim*len_data*len_data, n_class*len_data*len_data)
        self.n_class = n_class
        self.dim = dim

    def forward(self, x):
        h = x
        batch_size = len(x)
        h = h.view(-1, self.dim*len_data*len_data)
        h = self.linear(h)
        h = h.view(batch_size, self.n_class, len_data, len_data)
        return h



class MyDataSet(Dataset):
    def __init__(self, root=None, train=True, transform=None, input_width = 15, dim=2):
        self.root = root
        self.transform = transform
        mode = "train" if train else "test"
        self.input_width = input_width
        self.dim = dim
        data_dir = os.path.join(self.root, mode)
        self.all_data = glob.glob(data_dir + "/*")

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
        ####################
        lbl_data = np.where(lbl_data > 0, 1, lbl_data)
        ####################
        len_data = len(lbl_data)
        img = torch.zeros(self.dim, len(lbl_data), len(lbl_data))
        half = self.input_width//2
        for i in range(x - half, x + half + 1):
            for j in range(y - half, y + half + 1):
                if 0 <= i < len_data and 0 <= j < len_data:
                    img[0][i][j] = depth / 1000
                    for k in range(self.dim - 1):
                        img[k + 1][i][j] = mag**(k+1) / 10**(k+1)
        return img, lbl_data

def train(input_width, dim):
    gpu = 0
    batchsize = 5
    epoch = 20
    dataset = "../data"
    freq = -1
    out = "./result4_buffer"

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('')

    # Set up a neural network to train
    net = Linear(n_class=2, dim=dim)

    # Set model to GPU
    if gpu >= 0:
        # Make a specified GPU current
        print("GPU using")
        device = 'cuda:' + str(gpu)
        net = net.to(device)

    weights = torch.tensor(
        [1.0, 1.0])
    if gpu >= 0:
        weights = weights.to(device)
    # Setup a loss and an optimizer
    criterion = nn.CrossEntropyLoss(weight=weights)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9) #lr 0.001
    lr = 0.1
    optimizer = optim.Adam(net.parameters(), lr)

    # Load the data

    transform = transforms.Compose([transforms.ToTensor()])

    trainvalset = MyDataSet(root=dataset, train=True, transform=transform, input_width=input_width, dim=dim)
    # Split train/val
    n_samples = len(trainvalset)
    print("n_samples:", n_samples)
    trainsize = int(n_samples * 0.9)
    valsize = n_samples - trainsize
    trainset, valset = torch.utils.data.random_split(
        trainvalset, [trainsize, valsize])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize,
                                            shuffle=True, num_workers=2)
    # Setup result holder
    x = []
    train_loss_record = []
    val_loss_record = []
    # Train
    for ep in range(epoch):  # Loop over the dataset multiple times

        running_loss = 0.0
        val_loss = 0.0

        for s, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            if gpu >= 0:
                inputs = inputs.to(device)
                labels = labels.to(device)
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Add loss
            running_loss += loss.item()
            # print("trainloader_", s, "loss:", loss.item())

        # Report loss of the epoch
        print('[epoch %d] loss: %.3f' % (ep + 1, running_loss))

        # Save the model
        if (ep + 1) % freq == 0:
            path = out + "/model_" + str(ep + 1)
            torch.save(net.state_dict(), path)

        # Validation
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                if gpu >= 0:
                    images = images.to(device)
                    labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                # Add loss
                val_loss += loss.item()

        # Record result
        x.append(ep + 1)
        train_loss_record.append(running_loss)
        val_loss_record.append(val_loss)

    print('Finished Training')
    path = out + "/model_final"
    torch.save(net.state_dict(), path)

def test(input_width, dim):
    gpu = 0
    batchsize = 5
    dataset = "../data"
    model = "./result4_buffer/model_final"

    print('GPU: {}'.format(gpu))
    print('# Minibatch-size: {}'.format(batchsize))
    print('')

    # Set up a neural network to test
    net = Linear(n_class=2, dim=dim)
    # Load designated network weight
    net.load_state_dict(torch.load(model))
    # Set model to GPU
    if gpu >= 0:
        # Make a specified GPU current
        print("GPU using")
        device = 'cuda:' + str(gpu)
        net = net.to(device)

    # Load the data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = MyDataSet(root=dataset, train=False, transform=transform, input_width=input_width, dim=dim)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                             shuffle=False, num_workers=2)

    mask_data = np.loadtxt("./mask.csv", delimiter=',', dtype=int)
    mask_but_positive = 0

    # Test
    total = 0
    data_matrix = [[0. for _ in range(2)] for _ in range(2)]  #TP, FN, FP, TN
    predict_array = []
    label_array = []
    with torch.no_grad():
        for data in testloader:
            # Get the inputs; data is a list of [inputs, labels]
            images, labels = data
            if gpu >= 0:
                images = images.to(device)
                labels = labels.to(device)
            # Forward
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            # Check whether estimation is right
            # c = (predicted == labels).squeeze()
            for i in range(len(predicted)):
                for j in range(len(predicted[i])):
                    for k in range(len(predicted[i][j])):
                        total += 1
                        label = labels[i][j][k].item()
                        predict = predicted[i][j][k].item()
                        label_array.append(label)
                        predict_array.append(predict)
                        ############################
                        if label == 1 and predict == 1:
                            data_matrix[0][0] += 1
                        elif label == 1 and predict == 0:
                            data_matrix[0][1] += 1
                        elif label == 0 and predict == 1:
                            data_matrix[1][0] += 1
                        else:
                            data_matrix[1][1] += 1
                        if mask_data[j][k] == 0 and predict > 0:
                            mask_but_positive += 1

    # List of classes
    print("confusion_matrix:")
    print("TP: ", data_matrix[0][0], " FN: ", data_matrix[0][1])
    print("FP: ", data_matrix[1][0], " TN: ", data_matrix[1][1])
    TP = data_matrix[0][0]
    FN = data_matrix[0][1]
    FP = data_matrix[1][0]
    TN = data_matrix[1][1]
    print("")
    print("mask_but_positive: ", mask_but_positive)
    print("Accuracy: ", (TP + TN)/(TP + FP + TN + FN))
    print("Precision: ", TP/(TP + FP))
    print("Recall: ", TP/(TP + FN))
    print("Specificity: ", TN/(FP + TN))
    print("F-score: ", 2*TP/(2*TP + FP +FN))


def main():
    for i in range(5):
        for j in range(5):
            w = 11 + 2*i
            d = 7 + 2*j
            print("")
            # print("input width: 15, dim: 9")
            print("input width: ", w, "dim: ", d)
            print("")
            train(w, d)
            test(w, d)
            print("")

if __name__ == "__main__":
    main()