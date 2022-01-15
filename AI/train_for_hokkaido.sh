#!/bin/sh

MINIBATCH=100
EPOCH=100

FOR_START=0
FOR_END=64

for i in `seq ${FOR_START} ${FOR_END}`
do
    KERNEL_SIZE=`expr 2 \* ${i} + 1`
    PARTATION="================== ""${KERNEL_SIZE}"" ========================="
    echo "${PARTATION}"

    sbatch train_c1fc2D.sh ${KERNEL_SIZE}
done

<< COMENTOUT
    t_mask
    ================== 1 =========================
    Submitted batch job 653610
    ================== 3 =========================
    Submitted batch job 653611
    ================== 5 =========================
    Submitted batch job 653612
    ================== 7 =========================
    Submitted batch job 653613
    ================== 9 =========================
    Submitted batch job 653614
    ================== 11 =========================
    Submitted batch job 653615
    ================== 13 =========================
    Submitted batch job 653616
    ================== 15 =========================
    Submitted batch job 653617
    ================== 17 =========================
    Submitted batch job 653618
    ================== 19 =========================
    Submitted batch job 653619
    ================== 21 =========================
    Submitted batch job 653620
    ================== 23 =========================
    Submitted batch job 653621
    ================== 25 =========================
    Submitted batch job 653622
    ================== 27 =========================
    Submitted batch job 653623
    ================== 29 =========================
    Submitted batch job 653624
    ================== 31 =========================
    Submitted batch job 653625
    ================== 33 =========================
    Submitted batch job 653626
    ================== 35 =========================
    Submitted batch job 653627
    ================== 37 =========================
    Submitted batch job 653628
    ================== 39 =========================
    Submitted batch job 653629
    ================== 41 =========================
    Submitted batch job 653630
    ================== 43 =========================
    Submitted batch job 653631
    ================== 45 =========================
    Submitted batch job 653632
    ================== 47 =========================
    Submitted batch job 653633
    ================== 49 =========================
    Submitted batch job 653634
    ================== 51 =========================
    Submitted batch job 653635
    ================== 53 =========================
    Submitted batch job 653636
    ================== 55 =========================
    Submitted batch job 653637
    ================== 57 =========================
    Submitted batch job 653638
    ================== 59 =========================
    Submitted batch job 653639
    ================== 61 =========================
    Submitted batch job 653640
    ================== 63 =========================
    Submitted batch job 653641
    ================== 65 =========================
    Submitted batch job 653642
    ================== 67 =========================
    Submitted batch job 653643
    ================== 69 =========================
    Submitted batch job 653644
    ================== 71 =========================
    Submitted batch job 653645
    ================== 73 =========================
    Submitted batch job 653646
    ================== 75 =========================
    Submitted batch job 653647
    ================== 77 =========================
    Submitted batch job 653648
    ================== 79 =========================
    Submitted batch job 653649
    ================== 81 =========================
    Submitted batch job 653650
    ================== 83 =========================
    Submitted batch job 653651
    ================== 85 =========================
    Submitted batch job 653652
    ================== 87 =========================
    Submitted batch job 653653
    ================== 89 =========================
    Submitted batch job 653654
    ================== 91 =========================
    Submitted batch job 653655
    ================== 93 =========================
    Submitted batch job 653656
    ================== 95 =========================
    Submitted batch job 653657
    ================== 97 =========================
    Submitted batch job 653658
    ================== 99 =========================
    Submitted batch job 653659
    ================== 101 =========================
    Submitted batch job 653660
    ================== 103 =========================
    Submitted batch job 653661
    ================== 105 =========================
    Submitted batch job 653662
    ================== 107 =========================
    Submitted batch job 653663
    ================== 109 =========================
    Submitted batch job 653664
    ================== 111 =========================
    Submitted batch job 653665
    ================== 113 =========================
    Submitted batch job 653666
    ================== 115 =========================
    Submitted batch job 653667
    ================== 117 =========================
    Submitted batch job 653668
    ================== 119 =========================
    Submitted batch job 653669
    ================== 121 =========================
    Submitted batch job 653670
    ================== 123 =========================
    Submitted batch job 653671
    ================== 125 =========================
    Submitted batch job 653672
    ================== 127 =========================
    Submitted batch job 653673
    ================== 129 =========================
    Submitted batch job 653674
COMENTOUT

<< COMENTOUT
    mapmask
    ================== 1 =========================
    Submitted batch job 654827
    ================== 3 =========================
    Submitted batch job 654828
    ================== 5 =========================
    Submitted batch job 654829
    ================== 7 =========================
    Submitted batch job 654830
    ================== 9 =========================
    Submitted batch job 654831
    ================== 11 =========================
    Submitted batch job 654832
    ================== 13 =========================
    Submitted batch job 654833
    ================== 15 =========================
    Submitted batch job 654834
    ================== 17 =========================
    Submitted batch job 654835
    ================== 19 =========================
    Submitted batch job 654836
    ================== 21 =========================
    Submitted batch job 654837
    ================== 23 =========================
    Submitted batch job 654838
    ================== 25 =========================
    Submitted batch job 654839
    ================== 27 =========================
    Submitted batch job 654840
    ================== 29 =========================
    Submitted batch job 654841
    ================== 31 =========================
    Submitted batch job 654842
    ================== 33 =========================
    Submitted batch job 654843
    ================== 35 =========================
    Submitted batch job 654844
    ================== 37 =========================
    Submitted batch job 654845
    ================== 39 =========================
    Submitted batch job 654846
    ================== 41 =========================
    Submitted batch job 654847
    ================== 43 =========================
    Submitted batch job 654848
    ================== 45 =========================
    Submitted batch job 654849
    ================== 47 =========================
    Submitted batch job 654850
    ================== 49 =========================
    Submitted batch job 654851
    ================== 51 =========================
    Submitted batch job 654852
    ================== 53 =========================
    Submitted batch job 654853
    ================== 55 =========================
    Submitted batch job 654854
    ================== 57 =========================
    Submitted batch job 654855
    ================== 59 =========================
    Submitted batch job 654856
    ================== 61 =========================
    Submitted batch job 654857
    ================== 63 =========================
    Submitted batch job 654858
    ================== 65 =========================
    Submitted batch job 654859
    ================== 67 =========================
    Submitted batch job 654860
    ================== 69 =========================
    Submitted batch job 654861
    ================== 71 =========================
    Submitted batch job 654862
    ================== 73 =========================
    Submitted batch job 654863
    ================== 75 =========================
    Submitted batch job 654864
    ================== 77 =========================
    Submitted batch job 654865
    ================== 79 =========================
    Submitted batch job 654866
    ================== 81 =========================
    Submitted batch job 654867
    ================== 83 =========================
    Submitted batch job 654868
    ================== 85 =========================
    Submitted batch job 654869
    ================== 87 =========================
    Submitted batch job 654870
    ================== 89 =========================
    Submitted batch job 654871
    ================== 91 =========================
    Submitted batch job 654872
    ================== 93 =========================
    Submitted batch job 654873
    ================== 95 =========================
    Submitted batch job 654874
    ================== 97 =========================
    Submitted batch job 654875
    ================== 99 =========================
    Submitted batch job 654876
    ================== 101 =========================
    Submitted batch job 654877
    ================== 103 =========================
    Submitted batch job 654878
    ================== 105 =========================
    Submitted batch job 654879
    ================== 107 =========================
    Submitted batch job 654880
    ================== 109 =========================
    Submitted batch job 654881
    ================== 111 =========================
    Submitted batch job 654882
    ================== 113 =========================
    Submitted batch job 654883
    ================== 115 =========================
    Submitted batch job 654884
    ================== 117 =========================
    Submitted batch job 654885
    ================== 119 =========================
    Submitted batch job 654886
    ================== 121 =========================
    Submitted batch job 654887
    ================== 123 =========================
    Submitted batch job 654888
    ================== 125 =========================
    Submitted batch job 654889
    ================== 127 =========================
    Submitted batch job 654890
    ================== 129 =========================
    Submitted batch job 654891
COMENTOUT