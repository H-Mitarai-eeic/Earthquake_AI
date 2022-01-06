#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_ccfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_ccfc2D_6/ -b 100 -e 100

#650641→651824 data2000 exponent = 2 weight = (1, 0) ===== mask変更 ========= result_ccfc2D_1/
"""
予測震度と実際の震度のずれの分布
   -9 階級 :  0 % (total 0)
   -8 階級 :  0 % (total 0)
   -7 階級 :  0 % (total 0)
   -6 階級 :  0 % (total 10)
   -5 階級 :  0 % (total 26)
   -4 階級 :  0 % (total 66)
   -3 階級 :  0 % (total 344)
   -2 階級 :  0 % (total 2561)
   -1 階級 :  1 % (total 12858)
    0 階級 : 65 % (total 817703)
    1 階級 : 30 % (total 382897)
    2 階級 :  1 % (total 22283)
    3 階級 :  0 % (total 414)
    4 階級 :  0 % (total 30)
    5 階級 :  0 % (total 1)
    6 階級 :  0 % (total 1)
    7 階級 :  0 % (total 0)
    8 階級 :  0 % (total 0)
    9 階級 :  0 % (total 0)
matthews corrcoef 0.26414966505560716
"""
#以下ネットワーク間違い未修正
    #650642 data2000 exponent = 1 weight = (0.51, 0.49) ===== mask変更 ========= result_ccfc2D_2/
    #650643 data2000 exponent = 3 weight = (0.51, 0.49) ===== mask変更 ========= result_ccfc2D_3/
    #650644 data2000 exponent = 1 weight = (0.8, 0.2) ===== mask変更 ========= result_ccfc2D_4/
    #650645 data2000 exponent = 3 weight = (0.8, 0.2) ===== mask変更 ========= result_ccfc2D_5/

#651826 data2000 exponent = 2 weight = (1, 0) ===== mask変更 ========= result_ccfc2D_6/ reluあり
"""
予測震度と実際の震度のずれの分布
   -9 階級 :  0 % (total 0)
   -8 階級 :  0 % (total 0)
   -7 階級 :  0 % (total 0)
   -6 階級 :  0 % (total 15)
   -5 階級 :  0 % (total 37)
   -4 階級 :  0 % (total 80)
   -3 階級 :  0 % (total 343)
   -2 階級 :  0 % (total 2797)
   -1 階級 :  1 % (total 14744)
    0 階級 : 32 % (total 408162)
    1 階級 : 63 % (total 782186)
    2 階級 :  2 % (total 30358)
    3 階級 :  0 % (total 458)
    4 階級 :  0 % (total 14)
    5 階級 :  0 % (total 0)
    6 階級 :  0 % (total 0)
    7 階級 :  0 % (total 0)
    8 階級 :  0 % (total 0)
    9 階級 :  0 % (total 0)
matthews corrcoef 0.19253198487862722
"""