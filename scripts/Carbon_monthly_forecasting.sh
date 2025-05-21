comment='MOMENT-Carbon'
learning_rate=0.001

for seq_len in 2 8 14 20
do
  for pred_len in 1 2 4 6 8 10 12 14 16 18
  do
    for feats_sel in 0
    do
      python run_carbon.py \
        --task_name long-term-forecast \
        --root_path ./data/carbon/res/ \
        --data_path merged_data.csv \
        --results_path ./results/data_carbon_monthly/ \
        --model_id MOMENT \
        --data Carbon_monthly \
        --features MS \
        --feats_pct $feats_sel \
        --target 'Price' \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --no-scale \
        --learning_rate $learning_rate \
        --batch_size 16 \
        --train_epochs 50 \
        --gpu_id 1 \
        --des "carbon_monthly" \
        --model_comment $comment | tee results/log/MOMENT_Carbon_Monthly_${pred_len}.txt
    done
  done
done
