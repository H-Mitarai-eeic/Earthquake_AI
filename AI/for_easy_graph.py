import matplotlib.pyplot as plt
# fig = plt.figure(dpi=600)
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(x, train_loss_record, label="train", color="red")
# ax2 = ax1.twinx()
# ax2.plot(x, val_loss_record, label="validation", color="blue")

# h1, l1 = ax1.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax1.legend(h1+h2, l1+l2, loc='upper right')

# ax1.set_xlabel("Epoch")
# ax1.set_ylabel("Train Loss")
# ax2.set_ylabel("Validation Loss")

# plt.savefig(args.out + '/accuracy_earthquaker.png')


# for power score
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# y = [0.3992923732507801, 0.41654005956948953, 0.44327045531569287, 0.4642941322890607, 0.47830390324474853, 0.4947989835854527, 0.4925008205378504, 0.4998942628715685, 0.501053446011743, 0.4959125267667251, 0.49493911585647266]
# fig = plt.figure(dpi=600)
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(x, y, color="red")
# ax1.scatter(x, y, color="red")
# ax1.set_ylim(0.3,0.6)

# ax1.set_xlabel("Power")
# ax1.set_ylabel("MCC")

# plt.savefig("beki_score.png")

# for power score
x = [1, 5, 9, 11, 13, 15, 17, 21]
y = [0.25277950363931767, 0.4479699883864418, 0.48420837338286676, 0.5012602232978033, 0.5098710708948291, 0.5107514782823582, 0.5061152981459649, 0.503767887879515]
fig = plt.figure(dpi=600)
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x, y, color="red")
ax1.scatter(x, y, color = "red")
ax1.set_ylim(0.2,0.6)

ax1.set_xlabel("scale")
ax1.set_ylabel("MCC")

plt.savefig("input_width_score.png")