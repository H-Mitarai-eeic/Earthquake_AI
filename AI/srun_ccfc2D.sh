srun -p p -t 50:00 --gres=gpu:1 --pty python3 test_eq_ccfc2D.py -g 0 -b 100 -d data2000_honshu6464_InstrumentalIntensity/ -m result_ccfc2D_1/model_100