#!/bin/sh

MINIBATCH=100
EPOCH=100

DEGREE_MIN=0
DEGREE_MAX=20

for i in `seq ${DEGREE_MIN} ${DEGREE_MAX}`
do
    MAG_D=${i}
    DEPTH_D=1
    CROSS_D=0
    PARTATION="================== mag_d = ${MAG_D} depth_d = ${DEPTH_D} cross_d = ${CROSS_D} =========================="
    echo "${PARTATION}"

    sbatch train_polycfc2D.sh ${MAG_D} ${DEPTH_D} ${CROSS_D}
done

<< COMENTOUT
654635 ~ 654655 mag_d = 0~20 depth_d=1 cross_d = 0
COMENTOUT