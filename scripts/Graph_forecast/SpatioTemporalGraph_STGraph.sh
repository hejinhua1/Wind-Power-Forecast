export CUDA_VISIBLE_DEVICES=1

model_name=SpatioTemporalGraph

python -u run.py \
  --task_name Graph_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path data.feather \
  --model_id STGraph_96_96 \
  --model $model_name \
  --data STGraph \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 6 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


