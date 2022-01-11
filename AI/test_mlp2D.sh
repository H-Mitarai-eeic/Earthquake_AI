#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100

MODELROOT="result_mlp2D_1"
MODEL="/model_100"

DATA="data2000_honshu6464_InstrumentalIntensity/"
OUTROOT=${MODELROOT}

EXPAND=10

PARTATION="============================================"

echo ${PARTATION}

python3 test_eq_mlp2D.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -m ${MODELROOT}${MODEL} -o ${MODELROOT} -expand ${EXPAND}