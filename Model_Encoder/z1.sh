#!/bin/bash

#SBATCH -J z1
#SBATCH -o z1.out
#SBATCH -p gpu
#SBATCH --gres=gpu:1

export PATH=/home/xuyang/mambaforge/envs/ssl/bin:$PATH

python encoder.py --d_model 512 --nhead 8 --d_ff 2048 --num_layers 6 --dropout_rate 0.1 --feature_dim 128 --temperature 0.5 --da1 channel_random --da2 channel_random --batch_size 1024 --epoch 20 --learn_late 0.0001
python encoder.py --d_model 512 --nhead 8 --d_ff 2048 --num_layers 6 --dropout_rate 0.1 --feature_dim 128 --temperature 0.5 --da1 channel_random --da2 channel_resize --batch_size 1024 --epoch 20 --learn_late 0.0001
python encoder.py --d_model 512 --nhead 8 --d_ff 2048 --num_layers 6 --dropout_rate 0.1 --feature_dim 128 --temperature 0.5 --da1 channel_random --da2 add_noise      --batch_size 1024 --epoch 20 --learn_late 0.0001
python encoder.py --d_model 512 --nhead 8 --d_ff 2048 --num_layers 6 --dropout_rate 0.1 --feature_dim 128 --temperature 0.5 --da1 channel_random --da2 time_out       --batch_size 1024 --epoch 20 --learn_late 0.0001
python encoder.py --d_model 512 --nhead 8 --d_ff 2048 --num_layers 6 --dropout_rate 0.1 --feature_dim 128 --temperature 0.5 --da1 channel_random --da2 base_shift     --batch_size 1024 --epoch 20 --learn_late 0.0001
