#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_mlp2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_mlp2D_5/ -b 100 -e 100

#649663 myloss3 exponent=2 weight=(1, 0) 21*21
#649664 myloss3 exponent=1 weight=(0.51, 0.49) 21*21
#649665 myloss3 exponent=3 weight=(0.51, 0.49) 21*21
#649666 myloss3 exponent=1 weight=(0.8, 0.2) 21*21
#649667 myloss3 exponent=3 weight=(0.8, 0.2) 21*21