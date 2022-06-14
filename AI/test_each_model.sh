#!/bin/bash
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100
DATA="data/data_for_hokkaido_regression/"

KERNEL_SIZE=125 #125 15


MAG_D=14
DEPTH_D=14
CROSS_D=0
#MAG_D=13
#DEPTH_D=13
#CROSS_D=0


MODELROOT="results/model4osaka/"
MODEL_REG="/model_final_reg"
#MODEL_REG="/mapmask-model_final_reg"
#MODEL_CLS="/model_final_cls_binary"
MODEL_CLS="/model_final_cls_10class"
#MODEL_CLS="/model_final_5_7_cls"

#OUTROOT=${MODELROOT}
#OUTROOT="results/model4osaka/test_data_10class/"
OUTROOT="results/model4osaka/train_data_10class/"
#OUTROOT="results/model4osaka/mapmask-model_train_data/"
mkdir ${OUTROOT}

TRAIN=1

echo "MODEL: ${MODELROOT}${MODEL_REG} ${MODELROOT}${MODEL_CLS}"
echo "DATA: ${DATA}"
echo "MAG_DEGREE = ${MAG_D}"
echo "DEPTH_DEGREE = ${DEPTH_D}"
echo "CROSS_DEGREE = ${CROSS_D}"
echo "TRAIN = ${TRAIN}"

python3 test_each_model.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -mr ${MODELROOT}${MODEL_REG} -mc ${MODELROOT}${MODEL_CLS} -o ${OUTROOT} -kernel_size ${KERNEL_SIZE} -mag_d ${MAG_D} -depth_d ${DEPTH_D} -cross_d ${CROSS_D} -u ${TRAIN}

echo "done"
cat test_each_model.sh > ${OUTROOT}setting_for_likelihood.txt


<< COMENTOUT

COMENTOUT
