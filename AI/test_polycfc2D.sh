#!/bin/bash
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

GPU=0
MINIBATCH=100
DATA="data/data_for_hokkaido_regression/"

DEGREE_MIN=1
DEGREE_STEP=1
DEGREE_MAX=20

KERNEL_SIZE=15 #125

for i in `seq ${DEGREE_MIN} ${DEGREE_STEP} ${DEGREE_MAX}`
do
    MAG_D=17
    DEPTH_D=${i}
    CROSS_D=0

    #MODELROOT="results/result_polycfc2D_mapmask_mag_d${MAG_D}_depth_d${DEPTH_D}_cross_d${CROSS_D}_kernel${KERNEL_SIZE}/"
    MODELROOT="results/result_polycfc2D_mag_d${MAG_D}_depth_d${DEPTH_D}_cross_d${CROSS_D}_kernel${KERNEL_SIZE}/"
    #MODELROOT="result_polycfc2D_mag_d${MAG_D}_depth_d${DEPTH_D}_cross_d${CROSS_D}/"
    MODEL="/model_100"

    OUTROOT=${MODELROOT}

    PARTATION="======== mag_d = ${MAG_D} depth_d = ${DEPTH_D} cross_d = ${CROSS_D} kernel = ${KERNEL_SIZE} mapmask=========="

    echo "${PARTATION}"
    echo "MODEL: ${MODELROOT}${MODEL}"
    echo "DATA: ${DATA}"
    echo "MAG_DEGREE = ${MAG_D}"
    echo "DEPTH_DEGREE = ${DEPTH_D}"
    echo "CROSS_DEGREE = ${CROSS_D}"

    python3 test_eq_polycfc2D.py -g ${GPU} -b ${MINIBATCH} -d ${DATA} -m ${MODELROOT}${MODEL} -o ${OUTROOT} -kernel_size ${KERNEL_SIZE} -mag_d ${MAG_D} -depth_d ${DEPTH_D} -cross_d ${CROSS_D} >> Stat_data_for_hokkaido_mag_d17_depth_dn_cross_d0_kernelsize${KERNEL_SIZE}.csv
done

<< COMENTOUT
654746 mag_d 0 ~ 20 model_50 テストのテスト ←途中終了
654801 mag_d 0 ~ 20 model_100 kernelサイズミスで 129

655022 mag_d=0 ~ 20 depth_d=1 cross_d = 0 model_100 kenelsize=125のテスト
655038 mag_d=1~20 depth_d=1~20 cross_d = 0 model_100 kenelsize=125のテスト
655196 mag_d=1 depth_d=1~20 cross_d = 0 model_100 kenelsize=125のテスト
655225 mag_d=14 depth_d=14 cross_d = 2~20 model_100 kenelsize=125のテスト

655756 mag_d=1~20 depth_d=1~20 cross_d = 0 model_100 kenelsize=15のテスト mapmask
655315 mag_d=17 depth_d=1~20 cross_d = 0 model_100 kenelsize=125のテスト
COMENTOUT