import builtins
import random
bitSize = 256
# l = [[(i + j) / 256 for i in range(bitSize)] for j in range(bitSize)]
l = [[random.randint(-2, 11) for i in range(bitSize)] for j in range(bitSize)]

f = open('data.txt', 'w')
for i in range(bitSize):
  for j in range(bitSize):
    f.write(str(l[i][j]) + " ")
  f.write("\n")
f.close()
