srun -p p -t 50:00 --gres=gpu:1 --pty python3 test_eq_cfc.py -g 0 -d data1000_honshu6464_mag50/ -b 1 -m  result_cfc4/model_final -i 0000 -o data1000_honshu6464_mag50/