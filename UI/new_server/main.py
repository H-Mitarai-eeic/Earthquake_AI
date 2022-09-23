import argparse
from ast import arg
from distutils.log import debug
from time import time
from unittest import result
import torch

from mycfc2D import MYFCN
from Linear_binary import Linear

import csv
import datetime
import threading
from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler

import http.server as s
from urllib.parse import urlparse
from urllib.parse import parse_qs


debug = False

len_data = 64
input_width = 15
half = input_width // 2


def timer(txt):
    if debug:
        now = datetime.datetime.now()
        print(txt)
        print(now)
        print()


def InstrumentalIntensity2SesimicIntensity(II):
    if II < 0.5:
        return 0
    elif II < 1.5:
        return 1
    elif II < 2.5:
        return 2
    elif II < 3.5:
        return 3
    elif II < 4.5:
        return 4
    elif II < 5.0:
        return 5  # 5-
    elif II < 5.5:
        return 6  # 5+
    elif II < 6.0:
        return 7  # 6-
    elif II < 6.5:
        return 8  # 6+
    else:
        return 9  # 7


timer("started")
parser = argparse.ArgumentParser(description="Pytorch example: CIFAR-10")
parser.add_argument(
    "--batchsize",
    "-b",
    type=int,
    default=100,
    help="Number of images in each mini-batch",
)
parser.add_argument(
    "--model_kaiki",
    "-mk",
    default="model_final_reg",
    help="Path to the model for test",
)
parser.add_argument(
    "--model_Linear",
    "-ml",
    default="model_final_cls_10class",
    help="Path to the model for test",
)
parser.add_argument(
    "--output", "-o", default="data100/", help="Root directory of outputfile"
)
parser.add_argument("--x", "-x", default=52, help="Erthquake ID for input/output files")
parser.add_argument("--y", "-y", default=32, help="Erthquake ID for input/output files")
parser.add_argument(
    "--depth", "-depth", default=10, help="Erthquake ID for input/output files"
)
parser.add_argument(
    "--mag", "-mag", default=9.0, help="Erthquake ID for input/output files"
)
parser.add_argument(
    "--kernel_size", "-kernel_size", default=125, help="Root directory of dataset"
)
parser.add_argument(
    "--mag_degree", "-mag_d", default=14, help="Root directory of dataset"
)
parser.add_argument(
    "--depth_degree", "-depth_d", default=14, help="Root directory of dataset"
)
parser.add_argument(
    "--cross_degree", "-cross_d", default=0, help="Root directory of dataset"
)
args = parser.parse_args()
# Set up a neural network to test
mesh_size = (64, 64)
mag_degree = int(args.mag_degree)
depth_degree = int(args.depth_degree)
cross_degree = int(args.cross_degree)
data_channels = mag_degree + depth_degree + (cross_degree // 2) + 1
depth_max = 600
dim_cls = 2
kernel_size = int(args.kernel_size)
net = MYFCN(in_channels=data_channels, mesh_size=mesh_size, kernel_size=kernel_size)
net_Linear = Linear(n_class=10, dim=dim_cls)
# net_Linear = Linear(10)
# Load designated network weight


output_list = [-1, -1]


class Thread(threading.Thread):
    def __init__(self, thread_name, epicenter):
        self.thread_name = str(thread_name)
        self.epicenter = epicenter
        threading.Thread.__init__(self)

    def __str__(self):
        return self.thread_name

    def run(self):
        timer(self.thread_name + " started")
        if self.thread_name == "net":
            outputs = net(self.epicenter)
            output_list[0] = outputs
        elif self.thread_name == "net_Linear":
            outputs = net_Linear(self.epicenter)
            output_list[1] = outputs
        timer(self.thread_name + " ended")


def run():
    epicenter = torch.zeros(1, data_channels, mesh_size[1], mesh_size[0])
    epicenter_Linear = torch.zeros(1, dim_cls, mesh_size[1], mesh_size[0])
    x = int(args.x)
    y = int(args.y)
    depth = float(args.depth)
    mag = float(args.mag)

    print()
    print("(x: {}, y: {}, depth: {}, mag: {}) -> RUNNING".format(x, y, depth, mag))

    net.load_state_dict(torch.load(args.model_kaiki, map_location=torch.device("cpu")))
    net_Linear.load_state_dict(
        torch.load(args.model_Linear, map_location=torch.device("cpu"))
    )

    timer("main process started")
    i = 0
    epicenter[0][i][y][x] = 1
    i += 1
    for j in range(1, mag_degree + 1):
        epicenter[0][i][y][x] = (mag / 9) ** j
        i += 1
    for j in range(1, depth_degree + 1):
        epicenter[0][i][y][x] = (depth / depth_max) ** j
        i += 1
    for j in range(1, cross_degree // 2 + 1):
        epicenter[0][i][y][x] = ((mag / 9) * (depth / depth_max)) ** (2 * j)
        i += 1
    # koyu
    for k in range(int(args.x) - half, int(args.x) + half + 1):
        for j in range(int(args.y) - half, int(args.y) + half + 1):
            if 0 <= k < len_data and 0 <= j < len_data:
                epicenter_Linear[0][0][k][j] = float(args.depth) / 1000
                epicenter_Linear[0][1][k][j] = (float(args.mag) / 10) ** 9

    thread_name_list = ["net", "net_Linear"]
    thread_epicenter_list = [epicenter, epicenter_Linear]
    thread_list = []

    for i in range(len(thread_name_list)):
        thread = Thread(
            thread_name=thread_name_list[i], epicenter=thread_epicenter_list[i],
        )
        thread.start()
        thread_list.append(thread)
    for thread in thread_list:
        thread.join()

    # Forward
    outputs = output_list[0]
    outputs_Linear = output_list[1]
    # Predict the label

    predicted = [[0 for i in range(mesh_size[1])] for j in range(mesh_size[0])]
    _, predicted_Linear = torch.max(outputs_Linear, 1)
    print(predicted_Linear[0][0][0].item())
    for Y in range(mesh_size[1]):
        for X in range(mesh_size[0]):
            if predicted_Linear[0][Y][X].item() == 0:
                # predicted[Y][X] = 0
                predicted[Y][X] = InstrumentalIntensity2SesimicIntensity(
                    outputs[0][Y][X].item() - 0.30
                )
                # predicted[Y][X] = InstrumentalIntensity2SesimicIntensity(outputs[0][Y][X].item() - 0.59)
            else:
                predicted[Y][X] = InstrumentalIntensity2SesimicIntensity(
                    outputs[0][Y][X].item()
                )
    # csv出力
    with open("predicted_data.csv", "w") as fo:
        writer = csv.writer(fo, lineterminator=",")
        writer.writerows(predicted)


class MyHandler(s.BaseHTTPRequestHandler):
    def do_POST(self):
        self.make_data()

    def do_GET(self):
        self.make_data()

    def make_data(self):
        # urlパラメータを取得
        parsed = urlparse(self.path)
        # urlパラメータを解析
        params = parse_qs(parsed.query)

        args.x = params["x"][0]
        args.y = params["y"][0]
        args.mag = params["mag"][0]
        args.depth = params["depth"][0]

        run()

        f = open("predicted_data.csv", "r")
        data = f.read()
        f.close()
        print("prediction ended")
        # response
        body = data
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())


host = "0.0.0.0"
port = 8000
httpd = s.HTTPServer((host, port), MyHandler)
print("server started at PORT:%s" % port)
httpd.serve_forever()
