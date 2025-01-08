export CUDA_VISIBLE_DEVICES=1

model_name=KGraph

python -u run.py \
  --task_name KGraph_forecast \
  --is_training 1 \
  --root_path ./data/ \
  --data_path data_with_entity_id.feather \
  --model_id KGraph_96_96 \
  --model $model_name \
  --data KGraph \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


