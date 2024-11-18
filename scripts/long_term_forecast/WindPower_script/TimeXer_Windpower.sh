export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path full_data.feather \
  --model_id ElcPrice_24_24 \
  --model $model_name \
  --data ElcPrice \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 4 \
  --des 'exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path full_data.feather \
  --model_id ElcPrice_24_24 \
  --model $model_name \
  --data ElcPrice \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 8 \
  --des 'exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path full_data.feather \
  --model_id ElcPrice_24_24 \
  --model $model_name \
  --data ElcPrice \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 16 \
  --des 'exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path full_data.feather \
  --model_id ElcPrice_24_24 \
  --model $model_name \
  --data ElcPrice \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path full_data.feather \
  --model_id ElcPrice_24_24 \
  --model $model_name \
  --data ElcPrice \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 64 \
  --des 'exp' \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path full_data.feather \
  --model_id ElcPrice_24_24 \
  --model $model_name \
  --data ElcPrice \
  --features MS \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --batch_size 128 \
  --des 'exp' \
  --itr 1





