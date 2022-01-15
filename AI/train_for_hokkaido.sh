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
    PARTATION="======= mag_d = ${MAG_D} depth_d = ${DEPTH_D} cross_d = ${CROSS_D} =========="
    echo "${PARTATION}"

    sbatch train_polycfc2D.sh ${MAG_D} ${DEPTH_D} ${CROSS_D}
done

<< COMENTOUT
    654635 ~ 654655 mag_d = 0~20 depth_d=1 cross_d = 0
COMENTOUT
<< COMENTOUT
    kernel_size 125

    ======= mag_d = 0 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654755
    ======= mag_d = 1 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654756
    ======= mag_d = 2 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654757
    ======= mag_d = 3 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654758
    ======= mag_d = 4 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654759
    ======= mag_d = 5 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654760
    ======= mag_d = 6 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654761
    ======= mag_d = 7 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654762
    ======= mag_d = 8 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654763
    ======= mag_d = 9 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654764
    ======= mag_d = 10 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654765
    ======= mag_d = 11 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654766
    ======= mag_d = 12 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654767
    ======= mag_d = 13 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654768
    ======= mag_d = 14 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654769
    ======= mag_d = 15 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654770
    ======= mag_d = 16 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654771
    ======= mag_d = 17 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654772
    ======= mag_d = 18 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654773
    ======= mag_d = 19 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654774
    ======= mag_d = 20 depth_d = 1 cross_d = 0 ==========
    Submitted batch job 654775
COMENTOUT