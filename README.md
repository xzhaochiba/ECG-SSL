# ECG SSL  
Self-supervised ECG framework for differentiating takotsubo syndrome and ST elevation myocardial infarction.  

Installing the Python environment  
requirements.txt  

Run the code to train the model.  
```bash
python encoder.py --d_model 512 --nhead 8 --d_ff 2048 --num_layers 6 --dropout_rate 0.1 --feature_dim 128 --temperature 0.5 --da1 time_out --da2 time_out --batch_size 1024 --epoch 20 --learn_late 0.0001
```

Run the code for downstream tasks.
```bash
python downstream.py --k_folder 0 --d_model 512 --nhead 8 --d_ff 2048 --num_layers 6 --dropout_rate 0.1 --feature_dim 128 --model_path ../Model_Encoder/model_save/Encoder_512_8_2048_6_0.1_128_0.5_channel_random_add_noise_1024_20_0.0001_20.pth --batch_size 512 --epoch 200 --learn_late 0.0005  
```
