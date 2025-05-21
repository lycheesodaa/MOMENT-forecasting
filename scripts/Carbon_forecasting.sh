comment='MOMENT-Carbon'
learning_rate=0.001

for feats_sel in 0 30 50 75
do
#  for pred_len in 1 2 3 4 5 7 14 21 28 30 35 40 45 50 55 60 70 80 90
  for pred_len in 180 365 545
  do
    python run_carbon.py \
      --task_name long-term-forecast \
      --root_path ./data/carbon/res_daily/ \
      --data_path merged_data_imputed.csv \
      --results_path ./results/data_carbon/ \
      --model_id MOMENT \
      --data Carbon \
      --features MS \
      --feats_pct $feats_sel \
      --target 'Price' \
      --no-scale \
      --seq_len 512 \
      --label_len 0 \
      --pred_len $pred_len \
      --learning_rate $learning_rate \
      --batch_size 16 \
      --train_epochs 50 \
      --gpu_id 1 \
      --des 'carbon_daily' \
      --model_comment $comment | tee results/log/MOMENT_Carbon_${pred_len}.txt
  done
done
