export CUDA_VISIBLE_DEVICES=1

python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 4 \
  --learning_rate 0.0001


python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 8 \
  --learning_rate 0.0001


python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 16 \
  --learning_rate 0.0001



python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 32 \
  --learning_rate 0.0001


python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 64 \
  --learning_rate 0.0001


python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 128 \
  --learning_rate 0.0001


python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 256 \
  --learning_rate 0.0001



python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 16 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 512 \
  --learning_rate 0.0001


python -u /home/hjh/WindPowerForecast/exp/KGformer_normal_args.py \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --in_channels 26 \
  --hidden_channels 8 \
  --out_channels 1 \
  --timestep_max 96 \
  --nb_blocks 2 \
  --batch_size 32 \
  --learning_rate 0.0001







