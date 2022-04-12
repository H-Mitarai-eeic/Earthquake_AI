#!/bin/bash
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100
DATA="data/data_for_hokkaido_regression/"

DEGREE_MIN=1
DEGREE_STEP=1
DEGREE_MAX=20

KERNEL_SIZE=125 #125

#for i in `seq ${DEGREE_MIN} ${DEGREE_STEP} ${DEGREE_MAX}`
#do

MAG_D=14
DEPTH_D=14
CROSS_D=0

MERGE_OFFSET=1

#MODELROOT="results/result_polycfc2D_mapmask_mag_d${MAG_D}_depth_d${DEPTH_D}_cross_d${CROSS_D}_kernel${KERNEL_SIZE}/"
MODELROOT="results/model4osaka/"
#MODELROOT="result_polycfc2D_mag_d${MAG_D}_depth_d${DEPTH_D}_cross_d${CROSS_D}/"
MODEL_REG="/model_final_reg"
MODEL_CLS="/model_final_cls"

#OUTROOT=${MODELROOT}
OUTROOT="test_results/osaka_merge_offset${MERGE_OFFSET}/"
mkdir ${OUTROOT}

PARTATION="======== mag_d = ${MAG_D} depth_d = ${DEPTH_D} cross_d = ${CROSS_D} kernel = ${KERNEL_SIZE}=========="

echo "${PARTATION}"
echo "MODEL: ${MODELROOT}${MODEL_REG} ${MODELROOT}${MODEL_CLS}"
echo "DATA: ${DATA}"
echo "MAG_DEGREE = ${MAG_D}"
echo "DEPTH_DEGREE = ${DEPTH_D}"
echo "CROSS_DEGREE = ${CROSS_D}"
echo "MERGE_OFFSET = ${MERGE_OFFSET}"

python3 test_hybrid.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -mr ${MODELROOT}${MODEL_REG} -mc ${MODELROOT}${MODEL_CLS} -o ${OUTROOT} -kernel_size ${KERNEL_SIZE} -mag_d ${MAG_D} -depth_d ${DEPTH_D} -cross_d ${CROSS_D} -mo ${MERGE_OFFSET} >> ${OUTROOT}/Stat_data_for_osaka_mo${MERGE_OFFSET}.csv

echo "done"
cat test_hybrid.sh >> ${OUTROOT}setting.txt

<< COMENTOUT

COMENTOUT