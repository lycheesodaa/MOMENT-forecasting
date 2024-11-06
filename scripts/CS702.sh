comment='MOMENT-Demand'
learning_rate=0.001
gpu_id=0
moment_size='large'

python run_cs702.py \
  --task_name lp \
  --root_path ./data/dataset/ \
  --results_path ./results/cs702/ \
  --model_id MOMENT \
  --moment_size $moment_size \
  --data CS702 \
  --seq_len 10 \
  --label_len 0 \
  --pred_len 4 \
  --learning_rate $learning_rate \
  --percent 100 \
  --batch_size 8 \
  --gpu_id $gpu_id \
  --model_comment $comment | tee results/MOMENT_${pred_len}.txt