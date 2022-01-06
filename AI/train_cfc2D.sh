#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_cfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_cfc2D_12/ -b 100 -e 100

#645545→645620 data2000 myloss3 
#645621 data500
#645623 data2000 exponent = 3
#645624 data2000 exponent = 1 weight = (0.55, 0.45)
#645625 data2000 exponent = 1 weight = (0.7, 0.3)
#645626 data2000 exponent = 1 weight = (0.51, 0.49) ===== mask変更 =========

#ネットワーク修正
#646306 data2000 exponent = 1 weight = (0.51, 0.49) ===== mask変更 =========
#646319 data2000 exponent = 3 weight = (0.51, 0.49) ===== mask変更 =========
#646331 data2000 exponent = 1 weight = (0.8, 0.2) ===== mask変更 =========
#646334 data2000 exponent = 2 weight = (1, 0) ===== mask変更 =========  output:  result_cfc2D_10/
"""
予測震度と実際の震度のずれの分布
   -9 階級 :  0 % (total 0)
   -8 階級 :  0 % (total 0)
   -7 階級 :  0 % (total 0)
   -6 階級 :  0 % (total 4)
   -5 階級 :  0 % (total 34)
   -4 階級 :  0 % (total 80)
   -3 階級 :  0 % (total 319)
   -2 階級 :  0 % (total 2465)
   -1 階級 :  0 % (total 12274)
    0 階級 : 63 % (total 787171)
    1 階級 : 33 % (total 411093)
    2 階級 :  2 % (total 25129)
    3 階級 :  0 % (total 601)
    4 階級 :  0 % (total 23)
    5 階級 :  0 % (total 1)
    6 階級 :  0 % (total 0)
    7 階級 :  0 % (total 0)
    8 階級 :  0 % (total 0)
    9 階級 :  0 % (total 0)
matthews corrcoef 0.2577217060958623
"""
#646347 data2000 exponent = 3 weight = (0.8, 0.2) ===== mask変更 =========

#650667 data2000 exponent = 2 weight = (1, 0) ===== mask変更 ========= マグニチュード2乗    output:  result_cfc2D_12/
"""
予測震度と実際の震度のずれの分布
   -9 階級 :  0 % (total 0)
   -8 階級 :  0 % (total 0)
   -7 階級 :  0 % (total 0)
   -6 階級 :  0 % (total 0)
   -5 階級 :  0 % (total 3)
   -4 階級 :  0 % (total 44)
   -3 階級 :  0 % (total 177)
   -2 階級 :  0 % (total 1692)
   -1 階級 :  1 % (total 12865)
    0 階級 : 75 % (total 936442)
    1 階級 : 22 % (total 280388)
    2 階級 :  0 % (total 7107)
    3 階級 :  0 % (total 426)
    4 階級 :  0 % (total 43)
    5 階級 :  0 % (total 4)
    6 階級 :  0 % (total 0)
    7 階級 :  0 % (total 2)
    8 階級 :  0 % (total 0)
    9 階級 :  0 % (total 1)
matthews corrcoef 0.3296802279580119
"""