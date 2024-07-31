comment='MOMENT-Demand'

python run_demand.py \
  --task_name long_term_forecast \
  --root_path ./data/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id MOMENT \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --des 'Experiment' \
  --model_comment $comment | tee results/MOMENT_Demand_96.txt

python run_demand.py \
  --task_name long_term_forecast \
  --root_path ./data/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id MOMENT \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 192 \
  --des 'Experiment' \
  --model_comment $comment | tee results/MOMENT_Demand_192.txt

python run_demand.py \
  --task_name long_term_forecast \
  --root_path ./data/ \
  --data_path demand_data_all_cleaned.csv \
  --model_id MOMENT \
  --data Demand \
  --features MS \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 356 \
  --des 'Experiment' \
  --model_comment $comment | tee results/MOMENT_Demand_356.txt