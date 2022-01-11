#!/bin/sh

MINIBATCH=100
EPOCH=100

EXPAND_MIN=0
EXPAND_MAX=31

for i in `seq ${EXPAND_MIN} ${EXPAND_MAX}`
do
    EXPAND=${i}
    PARTATION="================== ""${EXPAND}"" =========================="
    echo "${PARTATION}"

    sbatch train_mlp2D.sh ${EXPAND}
done

<< COMENTOUT
    ================== 0 ==========================
    Submitted batch job 653530
    ================== 1 ==========================
    Submitted batch job 653531
    ================== 2 ==========================
    Submitted batch job 653532
    ================== 3 ==========================
    Submitted batch job 653533
    ================== 4 ==========================
    Submitted batch job 653534
    ================== 5 ==========================
    Submitted batch job 653535
    ================== 6 ==========================
    Submitted batch job 653536
    ================== 7 ==========================
    Submitted batch job 653537
    ================== 8 ==========================
    Submitted batch job 653538
    ================== 9 ==========================
    Submitted batch job 653539
    ================== 10 ==========================
    Submitted batch job 653540
    ================== 11 ==========================
    Submitted batch job 653541
    ================== 12 ==========================
    Submitted batch job 653542
    ================== 13 ==========================
    Submitted batch job 653543
    ================== 14 ==========================
    Submitted batch job 653544
    ================== 15 ==========================
    Submitted batch job 653545
    ================== 16 ==========================
    Submitted batch job 653546
    ================== 17 ==========================
    Submitted batch job 653547
    ================== 18 ==========================
    Submitted batch job 653548
    ================== 19 ==========================
    Submitted batch job 653549
    ================== 20 ==========================
    Submitted batch job 653550
    ================== 21 ==========================
    Submitted batch job 653551
    ================== 22 ==========================
    Submitted batch job 653552
    ================== 23 ==========================
    Submitted batch job 653553
    ================== 24 ==========================
    Submitted batch job 653554
    ================== 25 ==========================
    Submitted batch job 653555
    ================== 26 ==========================
    Submitted batch job 653556
    ================== 27 ==========================
    Submitted batch job 653557
    ================== 28 ==========================
    Submitted batch job 653558
    ================== 29 ==========================
    Submitted batch job 653559
    ================== 30 ==========================
    Submitted batch job 653560
    ================== 31 ==========================
    Submitted batch job 653561
COMENTOUT