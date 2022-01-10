#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_c1fc2_2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_c1fc2_2D_4/ -b 100 -e 100

#652637 result_c1fc2_2D_1/ dropout=False activation_flag = False
#652638 result_c1fc2_2D_2/ dropout=False activation_flag = True
#652639 result_c1fc2_2D_3/ dropout=True activation_flag = False
#652641 result_c1fc2_2D_4/ dropout=True activation_flag = True