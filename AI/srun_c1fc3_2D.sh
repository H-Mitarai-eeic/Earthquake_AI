#!/bin/sh

GPU=0
MINIBATCH=100
DATA="data2000_honshu6464_InstrumentalIntensity/"

MODELROOT="result_c1fc3_2D_2/"
MODEL="model_100"
OUT=${MODELROOT}

MINIBATCH=100

srun -p p -t 50:00 --gres=gpu:1 --pty python3 test_eq_c1fc3_2D.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -m ${MODELROOT}${MODEL} -o ${OUT}