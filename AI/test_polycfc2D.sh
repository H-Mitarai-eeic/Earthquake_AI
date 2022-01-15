#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100
DATA="data_for_hokkaido_regression/"

DEGREE_MIN=0
DEGREE_MAX=20

KERNEL_SIZE=129

for i in `seq ${DEGREE_MIN} ${DEGREE_MAX}`
do
    MAG_D=${i}
    DEPTH_D=1
    CROSS_D=0

    #MODELROOT="result_polycfc2D_mag_d${MAG_D}_depth_d${DEPTH_D}_cross_d${CROSS_D}_kernel${KERNEL_SIZE}/"
    MODELROOT="result_polycfc2D_mag_d${MAG_D}_depth_d${DEPTH_D}_cross_d${CROSS_D}/"
    MODEL="/model_100"

    OUTROOT=${MODELROOT}

    PARTATION="======== mag_d = ${MAG_D} depth_d = ${DEPTH_D} cross_d = ${CROSS_D} kernel = ${KERNEL_SIZE}=========="

    echo "${PARTATION}"
    echo "MODEL: ${MODELROOT}${MODEL}"
    echo "DATA: ${DATA}"
    echo "MAG_DEGREE = ${MAG_D}"
    echo "DEPTH_DEGREE = ${DEPTH_D}"
    echo "CROSS_DEGREE = ${CROSS_D}"

    python3 test_eq_polycfc2D.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -m ${MODELROOT}${MODEL} -o ${OUTROOT} -kernel_size ${KERNEL_SIZE} -mag_d ${MAG_D} -depth_d ${DEPTH_D} -cross_d ${CROSS_D} >> Stat_data_for_hokkaido_mag_d_kernelsize${KERNEL_SIZE}.csv
done

<< COMENTOUT
654746 mag_d 0 ~ 20 model_50 テストのテスト


COMENTOUT