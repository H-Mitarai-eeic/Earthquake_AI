import argparse
import torch
import numpy as np
from Linear import Linear
import csv
len_data = 64
# 予測プログラム


def main():
    parser = argparse.ArgumentParser(description='Earthquaker')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='./result/model_final',
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
    net = Linear(10)
    # Load designated network weight
    net.load_state_dict(torch.load(
        args.model, map_location=torch.device('cpu')))
    # Set model to GPU
    if args.gpu >= 0:
        # Make a specified GPU current
        print("GPU using")
        device = 'cuda:' + str(args.gpu)
        net = net.to(device)
    # Load the input
    inputs = torch.zeros(1, 2, len_data, len_data)
    for i in range(int(args.x)-10, int(args.x)+11):
        for j in range(int(args.y)-10, int(args.y)+11):
            if 0 <= i < len_data and 0 <= j < len_data:
                inputs[0][0][i][j] = float(args.depth) / 1000
                inputs[0][1][i][j] = float(args.magnitude) / 10

    if args.gpu >= 0:
        inputs = inputs.to(device)

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    pre_list = np.array(predicted).squeeze().tolist()
    with open('predicted_data.csv', 'w') as file:
        writer = csv.writer(file, lineterminator=',')
        writer.writerows(pre_list)
    print("predicted_data.csv is created!")


if __name__ == '__main__':
    main()
