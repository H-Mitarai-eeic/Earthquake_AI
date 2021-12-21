#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_fcdnn.py -g 0 -d data1000_honshu6464_mag50_InstrumentalIntensity/ -o result_fcdnn1/ -b 100 -e 100

#メモ
#638877 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -b 10 -o result_fcn1/ -b 100 -e 10000    new myfcn3 lr = 0.1
#638878 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -b 10 -o result_fcn2/ -b 100 -e 10000    new myfcn3 lr = 0.01
#638879 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -b 10 -o result_fcn3/ -b 100 -e 10000    new myfcn3 lr = 0.001
#　↑全0

#638973　python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -b 10 -o result_fcn1/ -b 100 -e 10000   new myfcn4 lr = 0.001
#638974　python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -b 10 -o result_fcn2/ -b 100 -e 10000   new myfcn4 lr = 0.01
#638975　python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -b 10 -o result_fcn3/ -b 100 -e 10000   new myfcn4 lr = 0.1

#線形回帰 fcのみ
#640471 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    lr = 0.1　マスクあり
#640472 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 0.01　マスクあり
#640479 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    lr = 1　マスクあり

#dataset.py変更
#640505 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    lr = 1　マスクあり
    #640506 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 0.1　マスクあり
#640579 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    lr = 1　マスクあり 4乗誤差

    #640601 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 1　マスクあり 4乗誤差, DNN
    #640622 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 1 マスクあり 四乗誤差　weight変更（0.5, 0.5）#悪くなし

    #640766 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 1 マスクあり 4乗誤差　weight変更（0.9, 0.1）   スパースでダメ
#640773 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn4/ -b 100 -e 100    lr = 1 マスクあり 4乗誤差　weight変更（0.5, 0.0, 0.5）
#640784 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 1 マスクあり 4乗誤差　weight変更（0.2, 0.0, 0.8）

#640795 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    lr = 1 マスクあり 4乗誤差　weight変更（0.2, 0.0, 0.8）DNN

#↑感度悪い
#dataset でマグニチュードを指数で与える
#641727 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    lr = 1 マスクあり 4乗誤差　weight変更（0.2, 0.0, 0.8）DNN
#↑誤差発散した

#dataset でマグニチュードを正規化せずに与える
#641734 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    lr = 1 マスクあり 2乗誤差　weight変更（0.2, 0.0, 0.8）DNN
#641733 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 1 マスクあり 2乗誤差　weight変更（0.2, 0.0, 0.8）SNN

#↑2乗誤差だと過小評価
#ReLU6関数導入
#641745 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    lr = 1 マスクあり 8乗誤差　weight変更（0.2, 0.0, 0.8）SNN
#641748 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    lr = 1 マスクあり 6乗誤差　weight変更（0.2, 0.0, 0.8）SNN
#641757 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    lr = 1 マスクあり 4乗誤差　weight変更（0.2, 0.0, 0.8）SNN

#↑ダメ
#1乗誤差導入ReLU6無し magの正規化、オフセットなし
#641900 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100   1乗誤差　マスクあり SNN
#641929 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    1乗誤差　マスクあり SNN
#641958 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    1乗誤差　マスクあり SNN
#641998 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn4/ -b 100 -e 100    3乗誤差　マスクあり SNN

#dataset.py修正
#642089 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    SNN
#642092 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    DNN
#642099 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    SNN?    
#642060 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn4/ -b 100 -e 100    DNN?

#SNN bais = False
#642767 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    SNN
#642774 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    SNN

#642803 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    SNN 新誤差関数
#642805 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn4/ -b 100 -e 100    SNN 新誤差関数

#データをベクトルから三次元テンソルに
#642824 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn4/ -b 100 -e 100    新誤差関数

#誤差関数をpoolのに戻した
#642862 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    畳み込み+fc
#642863 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    畳み込み+fc
#642877 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    畳み込み+fc lr0.01

#データ広げた
    #642887 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn1/ -b 100 -e 100    畳み込み+fc myloss1
#642890 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn2/ -b 100 -e 100    fc  myloss1
#642921 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn3/ -b 100 -e 100    fc  myloss1
    #642940 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn4/ -b 100 -e 100    畳み込み+fc myloss2
#642951 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50/ -o result_fcn5/ -b 100 -e 100    fc  myloss1

#↑畳み込み層
#計測震度
#643054 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50_InstulmentalIntensity/ -o result_fcn5/ -b 100 -e 100  SNN
    #643779 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50_InstulmentalIntensity/ -o result_fcn1/ -b 100 -e 100  SNN
#643783 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50_InstrumentalIntensity/ -o result_fcn1/ -b 100 -e 100  DNN
#643788/643789/643790/643793/643799　python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50_InstrumentalIntensity/ -o result_fcn4/ -b 10 -e 100 datasetのZを対数に

#643800 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50_InstrumentalIntensity/ -o result_fcn4/ -b 100 -e 100  DNN 
#643805 python3 train_eq_fcn.py -g 0 -d data1000_honshu6464_mag50_InstrumentalIntensity/ -o result_fcn1/ -b 100 -e 100 DNN 二乗誤差