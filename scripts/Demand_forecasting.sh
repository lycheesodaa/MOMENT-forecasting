comment='MOMENT-Demand'

for pred_len in 1 12
#for pred_len in 96 192 356
do
  python run_demand.py \
    --task_name forecasting \
    --root_path ./data/ \
    --data_path demand_data_all_cleaned.csv \
    --model_id MOMENT \
    --data Demand \
    --features MS \
    --seq_len 512 \
    --label_len 0 \
    --pred_len $pred_len \
    --des 'Experiment' \
    --model_comment $comment | tee results/MOMENT_Demand_${pred_len}.txt
done