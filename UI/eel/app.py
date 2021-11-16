# eelのインポート
import eel
import time
import ntplib
from time import ctime
from datetime import datetime
import random
import subprocess
bitSize = 256
import os
# これを書くことでJSからアクセスができます

path = '/Users/kakeru/Earthquake_AI/UI/eel'


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
    # command = ["python3", "predict_eq_myfcn.py", "-m", "result_myfcn/model_final", "-x", str(X), "-y", str(Y), "-depth", str(Depth), "-mag", str(Mag)]
    command = ["python3", path + "/web/python/test.py", str(X), str(Y), str(Depth), str(Mag)]
    proc = subprocess.Popen(command)  # ->コマンドが実行される(処理の終了は待たない)
    result = proc.communicate()
    """ 
    """
    f = open(path + '/web/data/predicted_data.csv', "r")
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
eel.start("html/index.html", size=(800, 1500))
