#!/bin/sh

MINIBATCH=100
EPOCH=100

DEGREE_MIN=5
DEGREE_STEP=1
DEGREE_MAX=20

for i in `seq ${DEGREE_MIN} ${DEGREE_STEP} ${DEGREE_MAX}`
do
    MAG_D=17
    DEPTH_D=${i}
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
    d 固定 m 増やす

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
<< COMENTOUT
    kernel_size 125
    一緒に増やす

    ======= mag_d = 2 depth_d = 2 cross_d = 0 ==========
    Submitted batch job 654917
    ======= mag_d = 3 depth_d = 3 cross_d = 0 ==========
    Submitted batch job 654918
    ======= mag_d = 4 depth_d = 4 cross_d = 0 ==========
    Submitted batch job 654919
    ======= mag_d = 5 depth_d = 5 cross_d = 0 ==========
    Submitted batch job 654920
    ======= mag_d = 6 depth_d = 6 cross_d = 0 ==========
    Submitted batch job 654921
    ======= mag_d = 7 depth_d = 7 cross_d = 0 ==========
    Submitted batch job 654922
    ======= mag_d = 8 depth_d = 8 cross_d = 0 ==========
    Submitted batch job 654923
    ======= mag_d = 9 depth_d = 9 cross_d = 0 ==========
    Submitted batch job 654924
    ======= mag_d = 10 depth_d = 10 cross_d = 0 ==========
    Submitted batch job 654925
    ======= mag_d = 11 depth_d = 11 cross_d = 0 ==========
    Submitted batch job 654926
    ======= mag_d = 12 depth_d = 12 cross_d = 0 ==========
    Submitted batch job 654927
    ======= mag_d = 13 depth_d = 13 cross_d = 0 ==========
    Submitted batch job 654928
    ======= mag_d = 14 depth_d = 14 cross_d = 0 ==========
    Submitted batch job 654929
    ======= mag_d = 15 depth_d = 15 cross_d = 0 ==========
    Submitted batch job 654930
    ======= mag_d = 16 depth_d = 16 cross_d = 0 ==========
    Submitted batch job 654931
    ======= mag_d = 17 depth_d = 17 cross_d = 0 ==========
    Submitted batch job 654932
    ======= mag_d = 18 depth_d = 18 cross_d = 0 ==========
    Submitted batch job 654933
    ======= mag_d = 19 depth_d = 19 cross_d = 0 ==========
    Submitted batch job 654934
    ======= mag_d = 20 depth_d = 20 cross_d = 0 ==========
    Submitted batch job 654935
COMENTOUT
<< COMENTOUT
    kernel_size 125
    m 固定 d 増やす  

    ======= mag_d = 1 depth_d = 2 cross_d = 0 ==========
    Submitted batch job 654936
    ======= mag_d = 1 depth_d = 3 cross_d = 0 ==========
    Submitted batch job 654937
    ======= mag_d = 1 depth_d = 4 cross_d = 0 ==========
    Submitted batch job 654938
    ======= mag_d = 1 depth_d = 5 cross_d = 0 ==========
    Submitted batch job 654939
    ======= mag_d = 1 depth_d = 6 cross_d = 0 ==========
    Submitted batch job 654940
    ======= mag_d = 1 depth_d = 7 cross_d = 0 ==========
    Submitted batch job 654941
    ======= mag_d = 1 depth_d = 8 cross_d = 0 ==========
    Submitted batch job 654942
    ======= mag_d = 1 depth_d = 9 cross_d = 0 ==========
    Submitted batch job 654943
    ======= mag_d = 1 depth_d = 10 cross_d = 0 ==========
    Submitted batch job 654944
    ======= mag_d = 1 depth_d = 11 cross_d = 0 ==========
    Submitted batch job 654945
    ======= mag_d = 1 depth_d = 12 cross_d = 0 ==========
    Submitted batch job 654946
    ======= mag_d = 1 depth_d = 13 cross_d = 0 ==========
    Submitted batch job 654947
    ======= mag_d = 1 depth_d = 14 cross_d = 0 ==========
    Submitted batch job 654948
    ======= mag_d = 1 depth_d = 15 cross_d = 0 ==========
    Submitted batch job 654949
    ======= mag_d = 1 depth_d = 16 cross_d = 0 ==========
    Submitted batch job 654950
    ======= mag_d = 1 depth_d = 17 cross_d = 0 ==========
    Submitted batch job 654951
    ======= mag_d = 1 depth_d = 18 cross_d = 0 ==========
    Submitted batch job 654952
    ======= mag_d = 1 depth_d = 19 cross_d = 0 ==========
    Submitted batch job 654953
    ======= mag_d = 1 depth_d = 20 cross_d = 0 ==========
    Submitted batch job 654954
COMENTOUT
<< COMENTOUT
    kernel 125
    mag_d14 depth_d 14 でcross_d 2~20

    ======= mag_d = 14 depth_d = 14 cross_d = 2 ==========
    Submitted batch job 655207
    ======= mag_d = 14 depth_d = 14 cross_d = 4 ==========
    Submitted batch job 655208
    ======= mag_d = 14 depth_d = 14 cross_d = 6 ==========
    Submitted batch job 655209
    ======= mag_d = 14 depth_d = 14 cross_d = 8 ==========
    Submitted batch job 655210
    ======= mag_d = 14 depth_d = 14 cross_d = 10 ==========
    Submitted batch job 655211
    ======= mag_d = 14 depth_d = 14 cross_d = 12 ==========
    Submitted batch job 655212
    ======= mag_d = 14 depth_d = 14 cross_d = 14 ==========
    Submitted batch job 655213
    ======= mag_d = 14 depth_d = 14 cross_d = 16 ==========
    Submitted batch job 655214
    ======= mag_d = 14 depth_d = 14 cross_d = 18 ==========
    Submitted batch job 655215
    ======= mag_d = 14 depth_d = 14 cross_d = 20 ==========
    Submitted batch job 655216
COMENTOUT
<<COMENTOUT
    kernel 125
    mag_d17 depth_d=2~20 でcross_d 0

    ======= mag_d = 17 depth_d = 2 cross_d = 0 ==========
    Submitted batch job 655226
    ======= mag_d = 17 depth_d = 3 cross_d = 0 ==========
    Submitted batch job 655227
    ======= mag_d = 17 depth_d = 4 cross_d = 0 ==========
    Submitted batch job 655228
    ======= mag_d = 17 depth_d = 5 cross_d = 0 ==========
    Submitted batch job 655245
    ======= mag_d = 17 depth_d = 6 cross_d = 0 ==========
    Submitted batch job 655246
    ======= mag_d = 17 depth_d = 7 cross_d = 0 ==========
    Submitted batch job 655247
    ======= mag_d = 17 depth_d = 8 cross_d = 0 ==========
    Submitted batch job 655248
    ======= mag_d = 17 depth_d = 9 cross_d = 0 ==========
    Submitted batch job 655249
    ======= mag_d = 17 depth_d = 10 cross_d = 0 ==========
    Submitted batch job 655250
    ======= mag_d = 17 depth_d = 11 cross_d = 0 ==========
    Submitted batch job 655251
    ======= mag_d = 17 depth_d = 12 cross_d = 0 ==========
    Submitted batch job 655252
    ======= mag_d = 17 depth_d = 13 cross_d = 0 ==========
    Submitted batch job 655253
    ======= mag_d = 17 depth_d = 14 cross_d = 0 ==========
    Submitted batch job 655254
    ======= mag_d = 17 depth_d = 15 cross_d = 0 ==========
    Submitted batch job 655255
    ======= mag_d = 17 depth_d = 16 cross_d = 0 ==========
    Submitted batch job 655256
    ======= mag_d = 17 depth_d = 17 cross_d = 0 ==========
    Submitted batch job 655257
    ======= mag_d = 17 depth_d = 18 cross_d = 0 ==========
    Submitted batch job 655258
    ======= mag_d = 17 depth_d = 19 cross_d = 0 ==========
    Submitted batch job 655259
    ======= mag_d = 17 depth_d = 20 cross_d = 0 ==========
    Submitted batch job 655260
COMENTOUT