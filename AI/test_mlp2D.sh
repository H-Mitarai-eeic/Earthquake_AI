#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100
DATA="data2000_honshu6464_InstrumentalIntensity/"

EXPAND_MIN=0
EXPAND_MAX=31

MODELROOT_START_NUM=6

for i in `seq ${EXPAND_MIN} ${EXPAND_MAX}`
do
    MODELROOT="result_mlp2D_"`expr ${i} + ${MODELROOT_START_NUM}`
    MODEL="/model_100"

    OUTROOT=${MODELROOT}

    EXPAND=${i}

    PARTATION="================== ""${EXPAND}"" =========================="

    echo "${PARTATION}"

    python3 test_eq_mlp2D.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -m ${MODELROOT}${MODEL} -o ${MODELROOT} -expand ${EXPAND}
done

#652803 expand 0 ~ 31のテスト