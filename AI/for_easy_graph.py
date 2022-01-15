import matplotlib.pyplot as plt
import numpy as np
from numpy import NaN
import seaborn as sns

# for power score
def power_score():
    y = [0.3901149270672358, 0.4206494917883885, 0.443427110687792, 0.4633526825916408, 0.4716882672848534, 0.4893202061996221, 0.49797765749125944, 0.4882725456705292, 0.5038806385169046, 0.4963224550832398, 0.48524314501494525, 0.4809697700033274, 0.4704099108613984, 0.450022867811469, 0.4431427523532218, 0.4439791723838553, 0.4209114682054451, 0.4133715924725581, 0.3876607524775294]
    x = []
    for i in range(len(y)):
        x.append(i+1)
    fig = plt.figure(dpi=600)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.axvline(x=9, ymax=0.75, color="black", linestyle="--")
    ax1.plot(x, y, color="red")
    ax1.scatter(x, y, color="red")
    ax1.set_ylim(0.35,0.55)

    ax1.set_xlabel("Power")
    ax1.set_ylabel("MCC")
    ax1.set_xticks(x)


# plt.savefig("beki_score.png")

# for width score
def width_score():
    y = [0.2540243229438158, 0.3918553184984967, 0.44397207466317234, 0.4786345515787697, 0.492388919069146, 0.5005019336059235, 0.5019833134734678, 0.5022279180557734, 0.5032771609154747, 0.49302877230210435, 0.4952131853909414, 0.4927019074865624, 0.4908287425379201, 0.5003634109828774, 0.48097260257399677, 0.4853474912822796, 0.4773669574578094, 0.47913835200922056, 0.47460869972396597, 0.4710616823866584]
    x = []
    for i in range(len(y)):
        x.append(i*2+1)
    fig = plt.figure(dpi=600)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.axvline(x=15, ymax=0.73, color="black", linestyle="--")
    ax1.plot(x, y, color="red")
    ax1.scatter(x, y, color = "red")
    ax1.set_ylim(0.2,0.6)


    ax1.set_xlabel("$k$")
    ax1.set_ylabel("MCC")
    ax1.set_xticks(x)

    plt.savefig("input_width_score.png")

def heatmap():
    fig = plt.figure(dpi=600)
    label = [9,8,7,6,5,4,3,2,1,0]
    matrix = [[1419635, 5356, 2215, 578, 105, 19, 2, 1, 0, 0], [5420, 3880, 2101, 336, 48, 5, 0, 1, 0, 0], [1871, 2213, 3519, 1049, 175, 16, 1, 2, 0, 0], [358, 501, 1443, 1247, 315, 47, 7, 3, 0, 0], [50, 123, 331, 513, 230, 38, 6, 1, 0, 0], [5, 12, 37, 83, 31, 4, 5, 2, 1, 0], [1, 4, 14, 25, 28, 4, 3, 3, 1, 0], [0, 2, 5, 20, 12, 5, 0, 0, 0, 0], [0, 1, 0, 5, 4, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]
    matrix = np.array(matrix)
    sm = matrix.sum(axis=1).reshape(10,1)
    matrix = matrix / sm
    matrix = np.flipud(matrix.T)
    arr_mask = (matrix < 0.01)
    sns.heatmap(matrix, vmax=1, vmin=0, cmap="jet", annot=True, linewidths=0.2, linecolor="black", yticklabels=label, fmt=".2f", mask=arr_mask, cbar_kws={'label': 'Ratio for each Intensity'})
    plt.xlabel("Intensity")
    plt.ylabel("Prediction")
    
    plt.tight_layout()
    plt.savefig("heatmap.png")

heatmap()
