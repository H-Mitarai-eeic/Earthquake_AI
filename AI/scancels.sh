#!/bin/sh

FS=654059
FE=654090

for i in `seq ${FS} ${FE}`
do
    scancel ${i}
done