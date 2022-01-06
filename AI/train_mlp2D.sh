#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_5/ -b 100 -e 100

#649663 myloss3 exponent=2 weight=(1, 0) 21*21  output:  result_mlp2D_1/
"""
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
matthews corrcoef 0.2510521150714836
"""
#649664 myloss3 exponent=1 weight=(0.51, 0.49) 21*21
#649665 myloss3 exponent=3 weight=(0.51, 0.49) 21*21
#649666 myloss3 exponent=1 weight=(0.8, 0.2) 21*21
#649667 myloss3 exponent=3 weight=(0.8, 0.2) 21*21