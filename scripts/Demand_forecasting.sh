comment='MOMENT-Demand'
learning_rate=0.001

#for pred_len in 1 12 72
for pred_len in 24 36 48 60
do
  python run_demand.py \
    --task_name long-term-forecast \
    --root_path ./data/ \
    --data_path demand_data_all_cleaned_numerical.csv \
    --results_path ./results/data/ \
    --model_id MOMENT \
    --data Demand \
    --features MS \
    --seq_len 512 \
    --label_len 0 \
    --pred_len $pred_len \
    --learning_rate $learning_rate \
    --batch_size 32 \
    --des 'Experiment' \
    --model_comment $comment | tee results/MOMENT_Demand_${pred_len}.txt
done