#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_gan.py -g 0 -d data1000_honshu6464_mag50_InstrumentalIntensity/ -o result_gan1/ -b 100 -e 100

#644988