#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_polycfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_polycfc2D_1/ -b 100 -e 100


#651975 data2000 exponent = 2 weight = (1, 0) ===== mask変更 =========  output:  result_polycfc2D_1/
"""
予測震度と実際の震度のずれの分布
   -9 階級 :  0 % (total 0)
   -8 階級 :  0 % (total 0)
   -7 階級 :  0 % (total 0)
   -6 階級 :  0 % (total 0)
   -5 階級 :  0 % (total 0)
   -4 階級 :  0 % (total 8)
   -3 階級 :  0 % (total 50)
   -2 階級 :  0 % (total 943)
   -1 階級 :  0 % (total 12353)
    0 階級 : 85 % (total 1058692)
    1 階級 : 12 % (total 159759)
    2 階級 :  0 % (total 6745)
    3 階級 :  0 % (total 539)
    4 階級 :  0 % (total 77)
    5 階級 :  0 % (total 24)
    6 階級 :  0 % (total 2)
    7 階級 :  0 % (total 1)
    8 階級 :  0 % (total 0)
    9 階級 :  0 % (total 1)
matthews corrcoef(マスクなし) 0.42052076390239257
決定係数 0.06653398161047208
自由度調整済み決定係数 0.06652494209790638
ピアソン相関係数 0.698681686275204
RSS 186510.50512105352
RSE 0.3879576108770094
"""
