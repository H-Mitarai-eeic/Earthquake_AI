import argparse
import torch
import numpy as np
from myfcn3 import MYFCN3
import csv
len_data = 64

def main():
    parser = argparse.ArgumentParser(description='Pytorch example: CIFAR-10')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='result/model_final',
						help='Path to the model for test')
    
    parser.add_argument('--x', '-x', default='0',
						help='x zahyou')
    parser.add_argument('--y', '-y', default='0',
						help='y zahyou')
    parser.add_argument('--depth', '-depth', default='10',
						help='depth of shingen')
    parser.add_argument('--magnitude', '-mag', default='7',
						help='magnitude of earthquake')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    
    print('')   
    # Set up a neural network to test
    net = MYFCN3(10)
    # Load designated network weight
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    # Set model to GPU
    if args.gpu >= 0:
    	# Make a specified GPU current
    	print("GPU using")
    	device = 'cuda:' + str(args.gpu)
    	net = net.to(device)    
    # Load the input
    inputs = torch.zeros(1, 2, len_data, len_data)
    print("x:", int(args.x))
    print("y:", int(args.y))
    print("depth:", args.depth)
    print("mag:", args.magnitude)
    for i in range(int(args.x)-10, int(args.x)+11):
        for j in range(int(args.y)-10, int(args.y)+11):
            print(i,j)
            if 0 <= i < len_data and 0 <= j < len_data:
                inputs[0][0][i][j] = float(args.depth) / 1000
                print("inputs:", inputs[0][0][i][j].item())
                inputs[0][1][i][j] = float(args.magnitude) / 10
    # inputs[0][0][int(args.x)][int(args.y)] = float(args.depth) / 1000
    # inputs[0][1][int(args.x)][int(args.y)] = float(args.magnitude) / 10
    print("inputs:", inputs[0])
    print("input max:", torch.max(inputs[0]).item())

    if args.gpu >= 0:
        inputs = inputs.to(device)

    outputs = net(inputs)
    print("outputs:", outputs[0])
    _, predicted = torch.max(outputs, 1)
    pre_list = np.array(predicted).squeeze().tolist()
    print("max:", max(list(map(lambda x: max(x), pre_list))))
    with open('predicted_data.csv', 'w') as file:
        writer = csv.writer(file, lineterminator=',')
        writer.writerows(pre_list)
    # print(pre_list)


if __name__ == '__main__':
	main()
