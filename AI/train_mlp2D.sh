#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0

MINIBATCH=100
EPOCH=100

EXPAND=${1}
DATA="data_for_hokkaido_regression/"
OUT="result_mlp2D_mapmask_expand_${EXPAND}/"

mkdir ${OUT}

python3 train_eq_mlp2D.py -g ${GPU} -d ${DATA} -o ${OUT} -b ${MINIBATCH} -e ${EPOCH} -expand ${EXPAND}

#649663 myloss3 exponent=2 weight=(1, 0) 21*21  output:  result_mlp2D_1/
<< COMMENTOUT
    予測震度と実際の震度のずれの分布
    -9 階級 :  0 % (total 0)
    -8 階級 :  0 % (total 0)
    -7 階級 :  0 % (total 0)
    -6 階級 :  0 % (total 0)
    -5 階級 :  0 % (total 20)
    -4 階級 :  0 % (total 59)
    -3 階級 :  0 % (total 225)
    -2 階級 :  0 % (total 1966)
    -1 階級 :  0 % (total 10365)
        0 階級 : 62 % (total 777068)
        1 階級 : 33 % (total 412108)
        2 階級 :  2 % (total 36705)
        3 階級 :  0 % (total 645)
        4 階級 :  0 % (total 31)
        5 階級 :  0 % (total 2)
        6 階級 :  0 % (total 0)
        7 階級 :  0 % (total 0)
        8 階級 :  0 % (total 0)
        9 階級 :  0 % (total 0)
    matthews corrcoef(マスクなし) 0.2510521150714836
    決定係数 -1.5927400338643412
    自由度調整済み決定係数 -1.5927442184331992
    ピアソン相関係数 0.5115963480149376
    RSS 518040.55406100914
    RSE 0.6465658377732413
COMMENTOUT

#649664 myloss3 exponent=1 weight=(0.51, 0.49) 21*21 output:  result_mlp2D_2/
#649665 myloss3 exponent=3 weight=(0.51, 0.49) 21*21 output:  result_mlp2D_3/
#649666 myloss3 exponent=1 weight=(0.8, 0.2) 21*21 output:  result_mlp2D_4/
#649667 myloss3 exponent=3 weight=(0.8, 0.2) 21*21 output:  result_mlp2D_5/

#652698 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_6/ -b 100 -e 100 -expand 0
#652699 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_7/ -b 100 -e 100 -expand 1
#652700 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_8/ -b 100 -e 100 -expand 2
#652701 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_9/ -b 100 -e 100 -expand 3
#652702 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_10/ -b 100 -e 100 -expand 4
#652703 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_11/ -b 100 -e 100 -expand 5
#652704 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_12/ -b 100 -e 100 -expand 6
#652706 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_13/ -b 100 -e 100 -expand 7
#652707 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_14/ -b 100 -e 100 -expand 8
#652708 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_15/ -b 100 -e 100 -expand 9
#652709 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_16/ -b 100 -e 100 -expand 10
#652710 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_17/ -b 100 -e 100 -expand 11
#652711 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_18/ -b 100 -e 100 -expand 12
#652712 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_19/ -b 100 -e 100 -expand 13
#652713 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_20/ -b 100 -e 100 -expand 14
#652714 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_21/ -b 100 -e 100 -expand 15
#652715 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_22/ -b 100 -e 100 -expand 16
#652716 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_23/ -b 100 -e 100 -expand 17
#652717 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_24/ -b 100 -e 100 -expand 18
#652718 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_25/ -b 100 -e 100 -expand 19
#652719 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_26/ -b 100 -e 100 -expand 20
#652720 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_27/ -b 100 -e 100 -expand 21
#652721 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_28/ -b 100 -e 100 -expand 22
#652722 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_29/ -b 100 -e 100 -expand 23
#652723 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_30/ -b 100 -e 100 -expand 24
#652724 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_31/ -b 100 -e 100 -expand 25
#652725 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_32/ -b 100 -e 100 -expand 26
#652726 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_33/ -b 100 -e 100 -expand 27
#652727 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_34/ -b 100 -e 100 -expand 28
#652728 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_35/ -b 100 -e 100 -expand 29
#652729 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_36/ -b 100 -e 100 -expand 30
#652730 python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_37/ -b 100 -e 100 -expand 31