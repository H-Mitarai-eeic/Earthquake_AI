#!/bin/sh

FS=654601
FE=654020

for i in `seq ${FS} ${FE}`
do
    scancel ${i}
done