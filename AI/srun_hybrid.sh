srun -p v -t 50:00 --gres=gpu:1 --pty python3 test_each_model.py -g 0 -o ./ -u 0