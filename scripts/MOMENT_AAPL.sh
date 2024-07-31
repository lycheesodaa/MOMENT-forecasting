comment='MOMENT-AAPL'

python run.py \
  --task_name long_term_forecast \
  --root_path ./data/top50_macro_processed/ \
  --data_path AAPL.csv \
  --model_id MOMENT \
  --data Stocks \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Experiment' \
  --model_comment $comment | tee results/MOMENT_AAPL.txt