comment='MOMENT-Demand'
learning_rate=0.001

python run_demand.py \
  --task_name long-term-forecast \
  --results_path ./results/cs702/ \
  --model_id MOMENT \
  --data Demand \
  --seq_len 10 \
  --label_len 0 \
  --pred_len 4 \
  --learning_rate $learning_rate \
  --batch_size 32 \
  --model_comment $comment | tee results/MOMENT_Demand_${pred_len}.txt