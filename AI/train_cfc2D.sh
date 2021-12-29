#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_cfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_cfc2D_1/ -b 100 -e 100

#645545 data2000 myloss3 