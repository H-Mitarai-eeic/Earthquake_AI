#!/bin/sh
#PATH="data1000_honshu6464_mag50/"
#ID="40278"
#python3 test_eq_fcn.py -g -1 -d ${PATH} -b 1 -m result_bin/model_1 -i ${ID} -o ${PATH}
#cat ${PATH}predicted/${ID}_predicted.csv
python3 test_eq_gan.py -g -1 -d data1000_honshu6464_mag50/ -b 1 -m result_bin/model_1 -i 933 -o data1000_honshu6464_mag50/
#cat data100_honshu6464/predicted/40278_predicted.csv | grep 0