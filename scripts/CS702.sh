comment='MOMENT-702'
learning_rate=0.0001
gpu_id=3
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
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --gpu_id $gpu_id \
  --model_comment $comment | tee results/MOMENT_${pred_len}.txt