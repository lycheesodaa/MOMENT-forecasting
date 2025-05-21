comment='MOMENT-Demand'
learning_rate=0.001

# only top0 will run zeroshot, since its the fastest
#for run_name in "top0" "top5" "top9" ""
for run_name in "featsel"
do
  # for pred_len in 1 12 24 48 72 168 336
  for pred_len in 1
  do
    python run_demand.py \
      --task_name long-term-forecast \
      --root_path ./data/ \
      --data_path demand_data_all_nsw_${run_name}.csv \
      --results_path ./results/data_aus/${run_name}/ \
      --model_id MOMENT \
      --data Demand \
      --features MS \
      --no-scale \
      --freq h \
      --seq_len 512 \
      --label_len 0 \
      --pred_len $pred_len \
      --learning_rate $learning_rate \
      --batch_size 16 \
      --gpu_id 1 \
      --des "aus_${run_name}" \
      --model_comment $comment 2>&1 | tee results/log/MOMENT_aus_Demand_${pred_len}_${run_name}.txt
  done
done
