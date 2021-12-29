#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_cfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_cfc2D_5/ -b 100 -e 100

#645545→645620 data2000 myloss3 
#645621 data500
#645623 data2000 exponent = 3
#645624 data2000 exponent = 1 weight = (0.55, 0.45)
#645624 data2000 exponent = 1 weight = (0.7, 0.3)