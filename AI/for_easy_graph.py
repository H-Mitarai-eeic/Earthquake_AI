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


plt.savefig("beki_score.png")

# for width score
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