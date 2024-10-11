comment='MOMENT-Demand'
learning_rate=0.001

#for pred_len in 1 12 72
for pred_len in 24 36 48 60
do
  CUDA_VISIBLE_DEVICES=1 python run_demand.py \
    --task_name long-term-forecast \
    --root_path ./data/ \
    --data_path demand_data_all_cleaned_top9.csv \
    --results_path ./results/feature_select_top9/ \
    --model_id MOMENT \
    --data Demand \
    --features MS \
    --seq_len 512 \
    --label_len 0 \
    --pred_len $pred_len \
    --learning_rate $learning_rate \
    --des 'top9' \
    --model_comment $comment | tee results/MOMENT_Demand_${pred_len}_top9.txt
done