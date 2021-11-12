from tkinter import*
import tkinter
# from tkinter.constants import FLAT
from PIL import Image, ImageTk
import random
import subprocess

x = 0
y = 0
depth = 0
mag = 0
figWidth = 512
bitSize = 256
gridSize = figWidth // bitSize

root = Tk()
root.title("EarthQuake")
root.geometry("800x1000")
bgcolor = "deep sky blue"
root.configure(bg=bgcolor)


labelTitle = tkinter.Label(text="Earthquake", fg="Black", font=("", "50", "bold"), bg=bgcolor)
labelTitle.pack(pady=10)

figJapan = "./fig/japan.png"
figJapan = Image.open(figJapan)
figJapan = figJapan.resize((figWidth, figWidth))
figJapan = ImageTk.PhotoImage(figJapan)

figRun = "./fig/run.png"
figRun = Image.open(figRun)
figRun = figRun.resize((300, 150))
figRun = ImageTk.PhotoImage(figRun)

canvas = Canvas(root, width=figWidth, height=figWidth)
canvas.create_image(0, 0, image=figJapan, anchor=tkinter.NW)


canvas.pack()


def circleclick(event):  # eventという機能を使います
  canvas.delete("all")
  canvas.create_image(0, 0, image=figJapan, anchor=tkinter.NW)
  x = event.x  # クリックした場所のx座標をxとする
  y = event.y  # クリックした場所のy座標をyとする

  #   cs = 10  # 円の大きさ
  f = open('dataMag.txt', 'r')
  mag = f.readline()
  f.close()
  # canvas.create_oval(x + cs, y + cs, x - cs, y - cs)  # クリックした場所に(楕)円
  canvas.create_text(x, y, text="❌", font=("HG丸ｺﾞｼｯｸM-PRO", int(50 * int(mag) / 12 + 10)))
  print(x, y, depth, mag)
  f = open('dataXY.txt', 'w')
  f.write(str(x) + " " + str(y))
  f.close()
  # writeData()


def changeDepth(str):
  depth = int(str)
  print(x, y, depth, mag)
  f = open('dataDepth.txt', 'w')
  f.write(str)
  f.close()


def changeMag(str):
  mag = int(str)
  print(x, y, depth, mag)
  f = open('dataMag.txt', 'w')
  f.write(str)
  f.close()


labelMag = tkinter.Label(text="マグニチュード", fg="Black", font=("", "20", "bold"), bg=bgcolor)
labelMag.pack()
canvas.bind("<Button-1>", circleclick)  # <Button-1>はマウスの左クリック


scaleMag = tkinter.Scale(
    root,
    orient=tkinter.HORIZONTAL,
    from_=-2,
    to=12,
    command=changeMag,
    # bg=bgcolor
)
scaleMag.pack(padx=100, pady=10, fill=tkinter.X)

labelDepth = tkinter.Label(text="深さ", fg="Black", font=("", "20", "bold"), bg=bgcolor)
labelDepth.pack()
scaleDepth = tkinter.Scale(
    root,
    orient=tkinter.HORIZONTAL,
    from_=0,
    to=1000,
    command=changeDepth,
    # bg=bgcolor
)
scaleDepth.pack(padx=100, pady=10, fill=tkinter.X)


def run():
  canvas.delete("all")
  canvas.create_image(0, 0, image=figJapan, anchor=tkinter.NW)
  color = ["#005FFF",
           "#136FFF",
           "#2C7CFF",
           "#4689FF",
           "#5D99FF",
           "#75A9FF",
           "#8EB8FF",
           "#A4C6FF",
           "#BAD3FF",
           "#D9E5FF",
           "#FFDBC9",
           "#FFC7AF",
           "#FFAD90",
           "#FF9872",
           "#FF8856",
           "#FF773E",
           "#FF6928",
           "#FF5F17",
           "#FF570D",
           "#FF4F02"]
  # l = [[(i + j) / 256 for i in range(bitSize)] for j in range(bitSize)]
  # l = [[random.randint(-2, 11) for i in range(bitSize)] for j in range(bitSize)]
  # print(l)

  command = ["python", "test.py"]
  proc = subprocess.Popen(command)  # ->コマンドが実行される(処理の終了は待たない)
  result = proc.communicate()
  l = [[0 for i in range(bitSize)] for j in range(bitSize)]
  f = open('data.txt', "r")

  for i in range(bitSize):
    line = f.readline()
    line = list(map(int, line.split()))
    l[i] = line
  for i in range(bitSize):
    for j in range(bitSize):
      if l[i][j] > 8:
        # if True:
        canvas.create_rectangle(gridSize * i, gridSize * j, gridSize * i + gridSize + 1, gridSize * j + gridSize, outline=color[l[i][j]], fill=color[l[i][j]])
      # canvas.create_text(gridSize * i, gridSize * j, text=str(int(l[i][j])))

    f = open('dataXY.txt', 'r')
    dataXY = f.readline()
    x, y = map(int, dataXY.split())
    f.close()
    f = open('dataMag.txt', 'r')
  mag = f.readline()
  f.close()
  # canvas.create_oval(x + cs, y + cs, x - cs, y - cs)  # クリックした場所に(楕)円
  canvas.create_text(x, y, text="❌", font=("HG丸ｺﾞｼｯｸM-PRO", int(50 * int(mag) / 12 + 10)))
  print(x, y, depth, mag)


buttonRun = tkinter.Button(
    root, text="RUN", command=run, font=("", "20"), image=figRun)
buttonRun.pack()

root.mainloop()
