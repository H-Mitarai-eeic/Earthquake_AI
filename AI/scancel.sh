#!/bin/sh

FS=654805
FE=654825

for i in `seq ${FS} ${FE}`
do
    scancel ${i}
done