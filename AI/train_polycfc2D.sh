#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_polycfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_polycfc2D_1/ -b 100 -e 100


#651975 data2000 exponent = 2 weight = (1, 0) ===== mask変更 =========  output:  result_polycfc2D_1/
