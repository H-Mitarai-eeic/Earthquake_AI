import csv
import random

for i in range(100):
    path = "data/Earthquake_random/train/random_sample_{}.csv".format(i)
    with open(path, "w") as f:
        writer = csv.writer(f)
        x = random.randrange(20,230)
        y = random.randrange(20,230)
        depth = random.randrange(10,300)
        mag = random.randrange(4, 8)
        arr = [[0 for _ in range(256)] for _ in range(256)]
        for i in range(256):
            for j in range(256):
                arr[i][j] = min(9, int(mag * 10 / (abs(x-i)/10 + 1) / (abs(y-j)/10 + 1)))
                # arr[i][j] = 1
        writer.writerow([x, y, depth, mag])
        writer.writerows(arr)