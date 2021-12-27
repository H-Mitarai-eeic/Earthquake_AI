#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_cfc.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_cfc7/ -b 100 -e 100

    #644688 cancel
    #644689 cancel
    #644690 cancel

    #644691 ネットワークミス
    #644692

# 644760 →　644763
#644764
#644765
# 644811　→ 644812 data all 4乗誤差
#644821 mylos3 result_cfc4 　←　良い
#645042 →　645043 mylos3 bias true　result_cfc5 　←　良くない

#645250 mylos3 result_cfc6 data500

#645277 myloss3 result_cfc7 data2000