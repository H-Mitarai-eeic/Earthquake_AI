#!/bin/sh
#SBATCH -p p
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 train_eq_ccfc2D.py -g 0 -d data2000_honshu6464_InstrumentalIntensity/ -o result_ccfc2D_6/ -b 100 -e 100

#650641→651824 data2000 exponent = 2 weight = (1, 0) ===== mask変更 ========= result_ccfc2D_1/
"""
予測震度と実際の震度のずれの分布
   -9 階級 :  0 % (total 0)
   -8 階級 :  0 % (total 0)
   -7 階級 :  0 % (total 1)
   -6 階級 :  0 % (total 19)
   -5 階級 :  0 % (total 45)
   -4 階級 :  0 % (total 107)
   -3 階級 :  0 % (total 649)
   -2 階級 :  0 % (total 3247)
   -1 階級 :  1 % (total 13790)
    0 階級 : 69 % (total 866958)
    1 階級 : 26 % (total 327793)
    2 階級 :  2 % (total 25881)
    3 階級 :  0 % (total 665)
    4 階級 :  0 % (total 31)
    5 階級 :  0 % (total 7)
    6 階級 :  0 % (total 0)
    7 階級 :  0 % (total 1)
    8 階級 :  0 % (total 0)
    9 階級 :  0 % (total 0)
matthews corrcoef(マスクなし) 0.2698231410568582
決定係数 -1.5627218096332665
自由度調整済み決定係数 -1.5627259457540252
ピアソン相関係数 0.43528951965864804
RSS 512042.784400583
RSE 0.6428120361384001
"""
#以下ネットワーク間違い未修正
    #650642 data2000 exponent = 1 weight = (0.51, 0.49) ===== mask変更 ========= result_ccfc2D_2/
    #650643 data2000 exponent = 3 weight = (0.51, 0.49) ===== mask変更 ========= result_ccfc2D_3/
    #650644 data2000 exponent = 1 weight = (0.8, 0.2) ===== mask変更 ========= result_ccfc2D_4/
    #650645 data2000 exponent = 3 weight = (0.8, 0.2) ===== mask変更 ========= result_ccfc2D_5/

#651826 data2000 exponent = 2 weight = (1, 0) ===== mask変更 ========= result_ccfc2D_6/ reluあり
"""
予測震度と実際の震度のずれの分布
   -9 階級 :  0 % (total 0)
   -8 階級 :  0 % (total 0)
   -7 階級 :  0 % (total 0)
   -6 階級 :  0 % (total 2)
   -5 階級 :  0 % (total 31)
   -4 階級 :  0 % (total 66)
   -3 階級 :  0 % (total 232)
   -2 階級 :  0 % (total 1873)
   -1 階級 :  0 % (total 10280)
    0 階級 : 27 % (total 341370)
    1 階級 : 65 % (total 810385)
    2 階級 :  5 % (total 72724)
    3 階級 :  0 % (total 2123)
    4 階級 :  0 % (total 97)
    5 階級 :  0 % (total 10)
    6 階級 :  0 % (total 0)
    7 階級 :  0 % (total 0)
    8 階級 :  0 % (total 1)
    9 階級 :  0 % (total 0)
matthews corrcoef(マスクなし) 0.178161315844874
決定係数 -3.8773102868605394
自由度調整済み決定係数 -3.8773181586257266
ピアソン相関係数 0.466875044859777
RSS 974507.4671319507
RSE 0.8867954585307902
"""