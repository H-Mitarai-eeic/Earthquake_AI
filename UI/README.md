# Earthquaker(HTTP通信バージョン)

Python eelではなく，純粋なサーバー(既定はlocalhost:3000)にクライアントがクエリを出す形で実行できる様にしました．

## 必要環境

* python3
* node.js

node.jsにexpress等が必要と言われた場合は `npm install hoge` で追加してあげてください．


## 実行方法


__アプリ実行に利用する学習済みデータ，「server/python/model_13」が重くてGitHubにあげられなかったため，再度学習して配置していただくか，[人工知能第二ターム共有用Googleドライブ](https://drive.google.com/file/d/1f4XASTDSxmUR9bWc5vx17uN6QSfRgP86/view?usp=sharing) から取得してserver/python/model_13と配置してください__

### サーバーの起動

serverディレクトリに移動して，`node main.js`としてください．
localhost:3000にサーバーが立ちます．

### クライアント側の接続


client/html/index.htmlを開いてくださればOKです．



## 注意

* 結果を保存ボタンがこの方式において使えなくなっています．修正を図ります．

* 実行が遅いのは通信部分ではなく，予測部分です．ここが応答速度のボトルネックになっているので，モデルの事前読み込み等で対応していきたいです．
