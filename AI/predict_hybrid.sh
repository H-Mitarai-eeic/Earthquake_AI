#python3 predict_eq_polycfc.py -x 38 -y 37 -d 14 -mag 6.8 -m result_polycfc2D_mag_d14_depth_d14_cross_d0_kernel125/model_100
python3 predict_eq_polycfc.py -x 52 -y 32 -d 24.4 -mag 9.0 -m results/result_polycfc2D_mag_d14_depth_d14_cross_d0_kernel125/model_100
#python3 predict_eq_polycfc.py -x 30 -y 30 -d 240 -mag 6.8 -m result_polycfc2D_mag_d14_depth_d14_cross_d0_kernel125/model_100
cat predicted_data_.csv | grep 0
