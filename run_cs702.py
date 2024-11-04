import argparse
import gc
from datetime import datetime, timedelta
import torch.nn.functional as F
import itertools
from momentfm import MOMENTPipeline
from torch.optim.lr_scheduler import OneCycleLR
from data_provider.data_factory import data_provider_cs702
from tqdm import tqdm
import random
import torch
import numpy as np
import os
import time
import pandas as pd

parser = argparse.ArgumentParser(description='MOMENT')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='MOMENT',
                    help='model name, options: [MOMENT, LSTM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='actual', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('--results_path', type=str, default='./results/data/')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--moment_size', type=str, default='large', choices=['small', 'base', 'large'])
parser.add_argument('--use_finetuned', type=bool, default=False)

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.MSELoss()
mae_metric = torch.nn.L1Loss()

target_index = 0  # this is the index of the actual target feature

setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
    args.task_name,
    args.model_id,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.des)

path = os.path.join(args.checkpoints,
                    setting + '-' + args.model_comment)

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

if not os.path.exists(args.results_path):
    os.makedirs(args.results_path)


def find_best_permutations_batched(correct_tensor, wrong_order_tensor):
    assert correct_tensor.dim() == 3 and wrong_order_tensor.dim() == 3, "Input tensors must be 3D"
    assert correct_tensor.shape == wrong_order_tensor.shape, f"Input tensors must have the same shape"

    batch_size, num_vectors, vector_dim = correct_tensor.shape
    permutations = list(itertools.permutations(range(num_vectors)))

    best_permutations = []

    for i in range(batch_size):
        best_permutation = None
        best_similarity = float('-inf')

        for perm in permutations:
            reordered_tensor = wrong_order_tensor[i, list(perm), :]
            similarity = F.cosine_similarity(correct_tensor[i], reordered_tensor)
            mean_similarity = similarity.mean().item()

            if mean_similarity > best_similarity:
                best_similarity = mean_similarity
                best_permutation = perm

        best_permutations.append(best_permutation)

    best_permutations_tensor = torch.tensor(best_permutations, dtype=torch.long, device=correct_tensor.device)
    assert best_permutations_tensor.shape == (batch_size, num_vectors)

    return best_permutations_tensor


def run_test(loader, output_csv=False, stage=None):
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            _batch_seq, _batch_cdd, _input_mask = [b.to(device) for b in batch]
            bsz, seq_len, n_feats = _batch_seq.shape

            _batch_seq = _batch_seq.permute(0, 2, 1).contiguous()
            _batch_seq = F.pad(_batch_seq, (0, 512 - seq_len), "constant", 0)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    _pred = model(_batch_seq, _input_mask)
            else:
                _pred = model(_batch_seq, _input_mask)

            _pred = _pred.permute(0, 2, 1).contiguous()
            pred_next_point = _pred[:, -1:, :].squeeze()
            _pred = _pred[:, :3, :]
            wrong_order = _batch_cdd.detach()

            best_perm_batched = find_best_permutations_batched(_pred, wrong_order)
            output_res = torch.cat((best_perm_batched, pred_next_point), dim=1)
            all_outputs.append(output_res.cpu())

    if output_csv:
        final_outputs = torch.cat(all_outputs, dim=0).numpy()
        df = pd.DataFrame(final_outputs)
        df[[0, 1, 2]] = df[[0, 1, 2]].astype(int)
        df.to_csv(os.path.join(args.results_path, f'predictions_epoch{args.train_epochs}.txt'),
                  sep=' ', index=False, header=False)

    print('Test complete.')


# Initialize model
model = MOMENTPipeline.from_pretrained(
    f"AutonLab/MOMENT-1-{args.moment_size}",
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': args.pred_len
    },
)
model.init()
model.to(device)

# Data loaders
train_data, train_loader = data_provider_cs702(args, 'full')
test_data, test_loader = data_provider_cs702(args, 'test')

# Optimizer and scheduler setup
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
max_lr = args.learning_rate
total_steps = len(train_loader)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=args.pct_start)

# Zero-shot eval only
if args.task_name == 'zero-shot':
    print(f'[DEBUG]: Forecasting for horizon length {args.pred_len}...')
    run_test(test_loader, True, 'zero-shot')
    print('Zero shot forecasting, ending inference...')
    exit()

# Load fine-tuned model if specified
if args.use_finetuned:
    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint')))
    torch.cuda.empty_cache()
    gc.collect()

    print(f'[DEBUG]: Fine-tuned eval for horizon length {args.pred_len}...')
    run_test(test_loader, True, 'lp')
    exit()

# Training
print(f'[DEBUG]: Training for horizon length {args.pred_len}...')
losses = []
start = time.time()
max_norm = 5.0  # For gradient clipping

for epoch in range(args.train_epochs):
    model.train()
    epoch_losses = []

    for batch in tqdm(train_loader, total=len(train_loader)):
        batch_seq, batch_cdd, batch_next_point, batch_labels, input_mask = [b.to(device) for b in batch]

        bsz, seq_len, n_feats = batch_seq.shape
        batch_seq = batch_seq.permute(0, 2, 1).contiguous()
        batch_seq = F.pad(batch_seq, (0, 512 - seq_len), "constant", 0)

        optimizer.zero_grad(set_to_none=True)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                pred = model(batch_seq, input_mask)
                pred = pred.permute(0, 2, 1).contiguous()
                true = torch.cat((batch_cdd, batch_next_point.unsqueeze(1)), dim=1)
                loss = criterion(pred, true)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(batch_seq, input_mask)
            pred = pred.permute(0, 2, 1).contiguous()
            true = torch.cat((batch_cdd, batch_next_point.unsqueeze(1)), dim=1)
            loss = criterion(pred, true)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        scheduler.step()
        epoch_losses.append(loss.item())

    epoch_mean_loss = np.mean(epoch_losses)
    losses.append(epoch_mean_loss)
    print(f"Epoch {epoch + 1}: Train loss: {epoch_mean_loss:.3f}\n")

elapsed = time.time() - start
average_loss = np.mean(losses)
print(f'Time elapsed: {elapsed}')
print(f"Total train loss: {average_loss:.3f}\n")

# Evaluation
print(f'[DEBUG]: Fine-tuned eval for horizon length {args.pred_len}...')
run_test(test_loader, True, 'lp')

# Save model
torch.save(model.state_dict(), os.path.join(path, 'checkpoint'))