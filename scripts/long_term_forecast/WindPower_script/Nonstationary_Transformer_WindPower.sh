export CUDA_VISIBLE_DEVICES=0

model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path data.feather \
  --model_id WindPower_96_96 \
  --model $model_name \
  --data WindPower \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 54 \
  --dec_in 54 \
  --c_out 54 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 128 \
  --batch_size 32
