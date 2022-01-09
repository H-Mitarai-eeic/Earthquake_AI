#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_c10fc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_c10fc2D_4/ -b 100 -e 100

#652407 result_c10fc2D_1/ dropout=False activation_flag = False
#652408 result_c10fc2D_2/ dropout=False activation_flag = True
#652409 result_c10fc2D_3/ dropout=True activation_flag = False
#652411 result_c10fc2D_3/ dropout=True activation_flag = True