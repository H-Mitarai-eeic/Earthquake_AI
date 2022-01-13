#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100
DATA="data_for_hokkaido_regression/"

EXPAND_MIN=0
EXPAND_MAX=1

MODELROOT_START_NUM=6

for i in `seq ${EXPAND_MIN} ${EXPAND_MAX}`
do
    #MODELROOT="result_mlp2D_"`expr ${i} + ${MODELROOT_START_NUM}`
    MODELROOT="result_mlp2D_expand_""${i}""/"
    MODEL="/model_100"

    OUTROOT=${MODELROOT}

    EXPAND=${i}

    PARTATION="================== ""${EXPAND}"" =========================="

    echo "${PARTATION}"
    echo "${MODELROOT}""${MODEL}"
    echo "${DATA}"
    echo "EXPAND = ""${EXPAND}"

    python3 test_eq_mlp2D.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -m ${MODELROOT}${MODEL} -o ${OUTROOT} -expand ${EXPAND} >> Stat_data_for_hokkaido.csv
done

#652838 expand 0 ~ 31のテスト