from collections import deque
# f = open("hoge.txt", "r")
d = deque()
with open("hoge.txt") as file:
  for line in file:
    print(line.rstrip())
# each line in f:
    line = line.split()
    print(line, "saa")
    d.appendleft(line[4])
    d.append(line[-1])
for i in range(len(d)):
  print(d[i], ",")
