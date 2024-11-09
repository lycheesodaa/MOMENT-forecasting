import argparse
from datetime import datetime

from momentfm import MOMENTPipeline
from torch.nn import DataParallel
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
import itertools

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
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='actual', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
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
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
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
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='number of gradient accumulation steps')

parser.add_argument('--results_path', type=str, default='./results/data/')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--moment_size', type=str, default='large', choices=['small', 'base', 'large'])
parser.add_argument('--use_finetuned', type=str)

args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(f'cuda:{args.gpu_id}')
criterion = torch.nn.MSELoss().to(device)
mae_metric = torch.nn.L1Loss().to(device)

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
                    setting + '-' + args.model_comment)  # unique checkpoint saving path

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

if not os.path.exists(args.results_path):
    os.mkdir(args.results_path)
    

def find_best_permutations_batched(correct_tensor, wrong_order_tensor):
    # Ensure tensors are 3D
    assert correct_tensor.dim() == 3 and wrong_order_tensor.dim() == 3, "Input tensors must be 3D"
    assert correct_tensor.shape == wrong_order_tensor.shape, f"Input tensors must have the same shape: {correct_tensor.shape} != {wrong_order_tensor.shape}"
    
    batch_size, num_vectors, vector_dim = correct_tensor.shape
    
    # Generate all possible permutations
    permutations = list(itertools.permutations(range(num_vectors)))
    
    best_permutations = []
    
    for i in range(batch_size):
        best_permutation = None
        best_similarity = float('-inf')
        
        for perm in permutations:
            # Reorder the wrong_order_tensor based on the current permutation
            reordered_tensor = wrong_order_tensor[i, list(perm), :]
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(correct_tensor[i], reordered_tensor)
            
            # Calculate the mean similarity across all vectors
            mean_similarity = similarity.mean().item()
            
            # Update if this permutation gives better similarity
            if mean_similarity > best_similarity:
                best_similarity = mean_similarity
                best_permutation = perm
        
        best_permutations.append(best_permutation)
    
    # Convert list of permutations to tensor
    best_permutations_tensor = torch.tensor(best_permutations, dtype=torch.long)

    assert best_permutations_tensor.shape == (batch_size, num_vectors)
    
    return best_permutations_tensor


def run_test(loader, output_csv=False, stage=None):
    all_outputs = []

    for batch in tqdm(loader, total=len(loader)):
        _batch_seq, _batch_cdd, _input_mask = [b.to(device) for b in batch]
        bsz, seq_len, n_feats = _batch_seq.shape

        _batch_seq = _batch_seq.permute(0, 2, 1).contiguous()
        _batch_seq = F.pad(_batch_seq, (512 - seq_len, 0), "constant", 0) # pad sequence since moment is fixed 512 input

        if args.use_amp:
            with torch.cuda.amp.autocast():
                _pred = model(_batch_seq, _input_mask) # this outputs an ndarray
        else:
            _pred = model(_batch_seq, _input_mask)  # this outputs an ndarray

        _pred = _pred.permute(0, 2, 1).contiguous()
        pred_next_point = _pred[:, -1:, :].squeeze().cpu() # should be (batch, num_features)
        _pred = _pred[:, :3, :]
        wrong_order = _batch_cdd.detach()

        best_perm_batched = find_best_permutations_batched(_pred, wrong_order) # should be (batch, 3)
        output_res = torch.cat((best_perm_batched, pred_next_point), dim=1) # should be (batch, 3 + num_features)

        all_outputs.append(output_res)

    if output_csv:
        # Concatenate all predictions and indices
        all_outputs = np.vstack([output.numpy() for output in all_outputs])

        df = pd.DataFrame(all_outputs)
        df[[0, 1, 2]] = df[[0, 1, 2]].astype(int)
        df.to_csv(args.results_path + f'predictions.txt', sep = ' ', index=False, header=False)

    print('Test complete.')


def create_datetime(row):
    row = row.astype(int)
    return datetime(row['year'], row['month'], row['day'], row['hour'], row['minute'])


# Forecasting task

model = MOMENTPipeline.from_pretrained(
    f"AutonLab/MOMENT-1-{args.moment_size}",
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': args.pred_len
    },
)
model.init()
model.to(device)

# train_data, train_loader = data_provider_cs702(args, 'train')
# vali_data, vali_loader = data_provider_cs702(args, 'val')
train_data, train_loader = data_provider_cs702(args, 'full')
test_data, test_loader = data_provider_cs702(args, 'test')

# Zero-shot eval only, if specified
if args.task_name == 'zero-shot':
    print(f'[DEBUG]: Forecasting for horizon length {args.pred_len}...')
    model.eval()
    with torch.no_grad():
        run_test(test_loader, True, 'zero-shot')
    model.train()

    print('Zero shot forecasting, ending inference...')
    exit()

if args.use_finetuned is not None:
    model.load_state_dict(torch.load(str(args.use_finetuned) + '/checkpoint'))
    torch.cuda.empty_cache()

    # ---- Evaluation only ----
    print(f'[DEBUG]: Fine-tuned eval for horizon length {args.pred_len}...')
    model.eval()
    with torch.no_grad():
        run_test(test_loader, True, 'lp')
    exit()


# ---- Training ----
print(f'[DEBUG]: Training for horizon length {args.pred_len}...')
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Create a OneCycleLR scheduler
max_lr = args.learning_rate
total_steps = len(train_loader) // args.gradient_accumulation_steps
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=args.pct_start)

# Gradient clipping value
max_norm = 5.0

cur_epoch = 0
max_epoch = 1

while cur_epoch < max_epoch:
    losses = []
    start = time.time()

    optimizer.zero_grad(set_to_none=True)
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_seq, batch_cdd, batch_next_point, batch_labels, input_mask = [b.to(device) for b in batch]

        bsz, seq_len, n_feats = batch_seq.shape
        batch_seq = batch_seq.permute(0, 2, 1).contiguous()
        batch_seq = F.pad(batch_seq, (512 - seq_len, 0), "constant", 0) # pad sequence since moment is fixed 512 input

        # train based on the last 4 points' loss (eval only shows the last point's loss instead)
        if args.use_amp:
            with torch.cuda.amp.autocast():
                pred = model(batch_seq, input_mask)  # this outputs an ndarray
                pred = pred.permute(0, 2, 1).contiguous() # (batch, pred_len, n_features)
                true = torch.cat((batch_cdd, batch_next_point.unsqueeze(1)), dim=1)

                assert true.shape == pred.shape

                train_loss = criterion(pred, true) / args.gradient_accumulation_steps

                # Scales the loss for mixed precision training
                scaler.scale(train_loss).backward()
        else:
            pred = model(batch_seq, input_mask)  # this outputs an ndarray
            pred = pred.permute(0, 2, 1).contiguous()  # (batch, pred_len, n_features)
            true = torch.cat((batch_cdd, batch_next_point.unsqueeze(1)), dim=1)

            assert true.shape == pred.shape

            train_loss = criterion(pred, true) / args.gradient_accumulation_steps

            train_loss.backward()

        losses.append(train_loss.item() * args.gradient_accumulation_steps)

        # Update weights if we've accumulated enough gradients
        if (idx + 1) % args.gradient_accumulation_steps == 0:
            if args.use_amp:
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

    elapsed = time.time() - start

    losses = np.array(losses)
    average_loss = np.average(losses)

    cur_epoch += 1
    print(f"Linear Probing Epoch {cur_epoch} - Train loss: {average_loss:.3f}\n")
    print(f'Time elapsed: {elapsed}')


# ---- Evaluation ----
print(f'[DEBUG]: Fine-tuned eval for horizon length {args.pred_len}...')
model.eval()
with torch.no_grad():
    run_test(test_loader, True, 'lp')

torch.save(model.state_dict(), str(path) + '/' + 'checkpoint')