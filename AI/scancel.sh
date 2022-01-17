#!/bin/sh

FS=655229
FE=655244

for i in `seq ${FS} ${FE}`
do
    scancel ${i}
done