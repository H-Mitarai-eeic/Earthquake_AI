#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_cfc.py -g 0 -d data_all_honshu6464_mag50_InstrumentalIntensity/ -o result_cfc3/ -b 100 -e 100

    #644688 cancel
    #644689 cancel
    #644690 cancel

    #644691 ネットワークミス
    #644692

# 644760 →　644763
#644764
#644765
# 644811 data all 4乗誤差