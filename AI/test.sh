#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:4
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#python3 train_eq_fcn.py -g 0 -d Earthquake_Data/data_reshaped/ -b 10
python3 test_eq_fcn.py -g 0 -d data100/ -b 1 -m result_fcn1/model_1 -i 40278 -o data100/