comment='MOMENT-Demand'
learning_rate=0.001
gpu_id=1
moment_size='small'

python run_cs702.py \
  --task_name long-term-forecast \
  --root_path ./data/dataset/ \
  --results_path ./results/cs702/ \
  --model_id MOMENT \
  --moment_size $moment_size \
  --data Demand \
  --seq_len 10 \
  --label_len 0 \
  --pred_len 4 \
  --learning_rate $learning_rate \
  --batch_size 16 \
  --gpu_id $gpu_id \
  --model_comment $comment | tee results/MOMENT_Demand_${pred_len}.txt