# eelのインポート
import eel
import time
import ntplib
from time import ctime
from datetime import datetime
import random
import subprocess
bitSize = 256

# これを書くことでJSからアクセスができます


@eel.expose
def ask_python_from_js_get_result(server, X, Y, Depth, Mag):
  # ここで処理を記述
  try:
    ntp_client = ntplib.NTPClient()
    ntp_resp = ntp_client.request(server)
  except:
    msg = "Woops! something went wrong."
  finally:
    # NOTE
    # return now_time
    # JSの関数を呼び出す
    command = ["python", "test.py", str(X), str(Y), str(Depth), str(Mag)]
    proc = subprocess.Popen(command)  # ->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()
    f = open('data.txt', "r")
    msg = ""
    for i in range(bitSize):
      line = f.readline()
      # line = list(map(int, line.split()))
      # l[i] = line
      msg += line
      msg += ","
    eel.run_js_from_python(msg)


# ウエブコンテンツを持つフォルダー
eel.init("web")

# 最初に表示するhtmlページ
eel.start("html/index.html", size=(800, 1300))
