#python3 predict_eq_polycfc.py -x 52 -y 32 -d 20 -mag 7 -m result_polycfc2D_1/model_17
python3 predict_eq_polycfc.py -x 60 -y 3 -d 200 -mag 7 -m result_polycfc2D_1/model_17
cat predicted_data_.csv | grep 0
