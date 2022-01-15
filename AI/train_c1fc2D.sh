#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0

MINIBATCH=100
EPOCH=100

KERNEL_SIZE=${1}
DATA="data_for_hokkaido_regression/"
#OUT="result_c1fc2D_karnel_""${KERNEL_SIZE}""/"
OUT="result_c1fc2D_mapmask_karnel_${KERNEL_SIZE}/"

mkdir ${OUT}
python3 train_eq_c1fc2D.py -g ${GPU} -d ${DATA} -o ${OUT} -b ${MINIBATCH} -e ${EPOCH} -kernel_size ${KERNEL_SIZE}

#652628 result_c1fc2D_1/ dropout=False activation_flag = False
#652629 result_c1fc2D_2/ dropout=False activation_flag = True
#652630 result_c1fc2D_3/ dropout=True activation_flag = False
#652631 result_c1fc2D_4/ dropout=True activation_flag = True