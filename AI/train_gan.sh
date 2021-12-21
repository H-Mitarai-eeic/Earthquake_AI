#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#python3 train_eq_fcn.py -g 0 -d Earthquake_Data/data_reshaped/ -b 10
python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000v2/ -b 100 -e 100000


#メモ
    #638056 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result/ -b 100

    #638174 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result/ -b 100 学習率0.1
    #638180 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result_bin/ -b 100 学習率0.001
    #638197 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result_bin/ -b 100 -e 500 学習率0.001

    #638282 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result_bin/ -b 100 -e 1000 
    #638284 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result/ -b 100 -e 1000    学習率0.001

    #638327 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000_honshu6464_mag50/ -b 100 -e 1000 学習率0.1

    #638386 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000_honshu6464_mag50/ -b 100 -e 1000 学習率0.001　データ正規化、Dの変更
        #638463 python3 train_eq_gan.py -g 0 -d data_all/ -o result1_all/ -b 2048 -e 10000  学習率0.001　データ正規化、Dの変更　resultフォルダミス
    #638485 python3 train_eq_gan.py -g 0 -d data_all/ -o result_all/ -b 2048 -e 10000       同上

    #638495 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result/ -b 100 -e 1000    学習率0.001　ノイズ無し
    #638496 python3 train_eq_gan.py -g 0 -d data_all/ -o result_bin/ -b 2048 -e 10000               学習率0.001　ノイズ無し

#predicted targets修正

    #638497 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000_honshu6464_mag50/ -b 100 -e 1000   学習率0.001 ノイズあり
    #638498 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result/ -b 100 -e 1000                        学習率0.001 ノイズ無し
    #638499 python3 train_eq_gan.py -g 0 -d data_all/ -o result_all/ -b 2048 -e 10000                                   学習率0.001 ノイズあり

    #638538 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result_bin/ -b 100 -e 100000  ノイズを10000で割った
    #638548 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000/ -b 100 -e 100000 ノイズを100で割った

    #638656 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000v2/ -b 100 -e 100000 ノイズを100で割った Dの正解ラベルのランダム化
    #638657 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000v2/ -b 100 -e 100000 ノイズを100で割った Dの正解ラベルのランダム化 改

#データセットのx, y修正

#638734 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result/ -b 100 -e 100000  学習率0.01 ノイズを100で割った Dの正解ラベルのランダム化 改
#638737 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000/ -b 100 -e 100000  学習率0.001 ノイズを100で割った Dの正解ラベルのランダム化 改
#638801 python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50/ -o result1000v2/ -b 100 -e 100000 学習率0.001 ノイズを100で割った Dの正解ラベルのランダム化してない
