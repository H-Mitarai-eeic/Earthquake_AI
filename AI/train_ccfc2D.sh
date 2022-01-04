#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_ccfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_ccfc2D_5/ -b 100 -e 100

#650641 data2000 exponent = 2 weight = (1, 0) ===== mask変更 ========= result_ccfc2D_1/
#650642 data2000 exponent = 1 weight = (0.51, 0.49) ===== mask変更 ========= result_ccfc2D_2/
#650643 data2000 exponent = 3 weight = (0.51, 0.49) ===== mask変更 ========= result_ccfc2D_3/
#650644 data2000 exponent = 1 weight = (0.8, 0.2) ===== mask変更 ========= result_ccfc2D_4/
#650645 data2000 exponent = 3 weight = (0.8, 0.2) ===== mask変更 ========= result_ccfc2D_5/