import builtins
import random
import sys

args = sys.argv
bitSize = 256
# l = [[(i + j) / 256 for i in range(bitSize)] for j in range(bitSize)]
l = [[random.randint(-2, 11) for i in range(bitSize)] for j in range(bitSize)]
print(args[1], args[2], args[3], args[4])
f = open('./web/data/data.txt', 'w')
for i in range(bitSize):
  for j in range(bitSize):
    f.write(str(l[i][j]) + ",")
  # f.write("\n")
f.close()
