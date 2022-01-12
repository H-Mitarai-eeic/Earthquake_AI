#!/bin/sh

HEAD="./result_c1fc2D_karnel_"
#HEAD="./result_mlp2D_"
for i in `seq 51 89`
do
    LOSSFILE="${HEAD}""${i}""LOSS.csv"
    MEANFILE="${HEAD}""${i}""Mean_error.csv"
    VFILE="${HEAD}""${i}""Variance_of_Error.csv"

    DIRNAME="${HEAD}""${i}""/"

    mv ${LOSSFILE} ${MEANFILE} ${VFILE} ${DIRNAME}

    mv "${DIRNAME}""${LOSSFILE}" "${DIRNAME}""LOSS.csv"  
    mv "${DIRNAME}""${MEANFILE}" "${DIRNAME}""Mean_error.csv"  
    mv "${DIRNAME}""${VFILE}" "${DIRNAME}""Variance_of_Error.csv"  
done