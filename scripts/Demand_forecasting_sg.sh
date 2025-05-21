comment='MOMENT-Demand'
learning_rate=0.001

# only top0 will run zeroshot, since its the fastest
#for run_name in "top0" "top5" "top9" ""
# for run_name in "featsel"
for run_name in "met"
do
  # for pred_len in 1 12 24 48 72 168 336
  for pred_len in 1 12 72 
  do
    python run_demand.py \
      --task_name long-term-forecast \
      --root_path ./data/ \
      --data_path demand_data_all_cleaned_${run_name}.csv \
      --results_path ./results/data/${run_name}/ \
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
      --gpu_id 0 \
      --des "sg_${run_name}" \
      --model_comment $comment 2>&1 | tee results/log/MOMENT_aus_Demand_${pred_len}_${run_name}.txt
  done
done