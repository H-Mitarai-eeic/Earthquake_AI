import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import matthews_corrcoef

from dataset import MyDataSet
from Linear import Linear


def main():
    parser = argparse.ArgumentParser(description='Earthquaker')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='../result/model_final',
                        help='Path to the model for test')
    parser.add_argument('--dataset', '-d', default='../data',
                        help='Root directory of dataset')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    # Set up a neural network to test
    net = Linear(10)
    # Load designated network weight
    net.load_state_dict(torch.load(args.model))
    # Set model to GPU
    if args.gpu >= 0:
        # Make a specified GPU current
        print("GPU using")
        device = 'cuda:' + str(args.gpu)
        net = net.to(device)

    # Load the data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = MyDataSet(root=args.dataset, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize,
                                             shuffle=False, num_workers=2)

    # Test
    total = 0
    data_matrix = [[0. for _ in range(3)] for _ in range(10)]  # TP, FN, FP
    predict_array = []
    label_array = []
    with torch.no_grad():
        for data in testloader:
            # Get the inputs; data is a list of [inputs, labels]
            images, labels = data
            if args.gpu >= 0:
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
                        if label == predict:
                            data_matrix[label][0] += 1
                        else:
                            data_matrix[label][1] += 1
                            data_matrix[predict][2] += 1

    # List of classes
    classes = ("0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7")
    
    # Show accuracy
    for i in range(10):
        tp = data_matrix[i][0]
        fn = data_matrix[i][1]
        fp = data_matrix[i][2]
        tn = total - tp - fn - fp
        class_total = tp + fn
        if class_total != 0 and tp + fn != 0 and tp + fp != 0:
            print('Class : %5s, Recall : %.2f, Precision : %.2f, Accuracy : %.2f, total num of this class: %d' % (
                classes[i], tp/(tp+fn), tp/(tp+fp), (tp+tn)/total, class_total))
                
    print(data_matrix)
    # print(label_array)
    # print(predict_array)
    print("matthews corrcoef", matthews_corrcoef(np.array(label_array), np.array(predict_array)))


if __name__ == '__main__':
    main()
