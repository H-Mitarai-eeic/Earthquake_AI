#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100
DATA="data_for_hokkaido_regression/"

FOR_START=0
FOR_END=64 #64

for i in `seq ${FOR_START} ${FOR_END}`
do
    KERNEL_SIZE=`expr 2 \* ${i} + 1`
    MODELROOT="result_c1fc2D_karnel_""${KERNEL_SIZE}""/"
    MODEL="/model_100"

    OUTROOT=${MODELROOT}

    PARTATION="================== ""${KERNEL_SIZE}"" =========================="

    echo "${PARTATION}"
    echo "${MODELROOT}""${MODEL}"
    echo "${DATA}"
    echo "KERNEL = ""${KERNEL_SIZE}"

    python3 test_eq_c1fc2D.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -m ${MODELROOT}${MODEL} -o ${OUTROOT} -kernel_size ${KERNEL_SIZE} >> Stat_data_for_hokkaido.csv
done

#654325 FOR_START=0 FOR_END=64 kernel_size 1~129
#654326