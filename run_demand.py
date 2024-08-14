import argparse
from datetime import datetime

from momentfm import MOMENTPipeline
from torch.nn import DataParallel
from torch.optim.lr_scheduler import OneCycleLR

from data_provider.data_factory import data_provider
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

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:1')
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
    os.mkdir(path)

is_zero_shot_short_horizon = args.pred_len <= 12

'''
For the demand forecasting task, the forecast data from NEMS should also be returned by the data loader.
Returning the data through the data loader helps to provide consistent evaluations of average loss across tests.
'''
def val_or_test(loader, output_csv=False, stage=None):
    total_loss = []
    total_mae_loss = []
    total_forecast_loss = []
    total_forecast_mae_loss = []
    all_preds = []
    all_true = []
    dates = []
    all_nems_forecasts = []

    for (batch_x, batch_y, batch_x_mark, batch_y_mark,
         nems_forecast_x, nems_forecast_y, input_mask) in tqdm(loader, total=len(loader)):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        nems_forecast_x = nems_forecast_x.to(device)
        nems_forecast_y = nems_forecast_y.to(device)
        input_mask = input_mask.to(device)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                pred = model(batch_x, input_mask) # this outputs an ndarray
        else:
            pred = model(batch_x, input_mask)  # this outputs an ndarray

        # pred = torch.from_numpy(pred)
        pred = pred[:, target_index, :]
        true = batch_y.detach()[:, target_index, :]

        loss = criterion(pred, true)
        mae_loss = mae_metric(pred, true)

        total_loss.append(loss.item())
        total_mae_loss.append(mae_loss.item())


        # get the loss values of NEMS' forecasts (this is only one column)
        forecasts = nems_forecast_y.detach()

        loss = criterion(forecasts, true)
        mae_loss = mae_metric(forecasts, true)

        total_forecast_loss.append(loss.item())
        total_forecast_mae_loss.append(mae_loss.item())

        if output_csv:
            all_preds.append(pred)
            all_true.append(true)
            dates.append(batch_y_mark.detach()[:, :, -args.pred_len:].permute(0, 2, 1).contiguous().view(-1, 6))
            all_nems_forecasts.append(forecasts)

    if output_csv:
        # Concatenate all predictions and indices
        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)
        dates = torch.cat(dates)
        all_nems_forecasts = torch.cat(all_nems_forecasts)

        # shape (12176, 96)
        all_preds = all_preds.cpu().float().numpy()
        all_true = all_true.cpu().float().numpy()
        dates = dates.cpu().numpy()
        all_nems_forecasts = all_nems_forecasts.cpu().float().numpy()

        # inverse the scaling
        all_preds = vali_data.target_inverse_transform(all_preds.reshape(-1, 1)).reshape(-1)
        all_true = vali_data.target_inverse_transform(all_true.reshape(-1, 1)).reshape(-1)
        all_nems_forecasts = vali_data.nems_inverse_transform(all_nems_forecasts.reshape(-1, 1)).reshape(-1)

        dates = pd.DataFrame(dates, columns=['year', 'month', 'day', 'weekday', 'hour', 'minute'], dtype=int)
        df = pd.DataFrame({
            'pred': all_preds,
            'true': all_true,
            'nems_forecast': all_nems_forecasts
        })
        df = pd.concat([dates, df], axis=1)
        df['date'] = df.apply(create_datetime, axis=1)
        df.drop(columns=['year', 'month', 'day', 'weekday', 'hour', 'minute'], inplace=True)
        df.to_csv(f'results/data/MOMENT_Demand_pl{args.pred_len}_{stage}_predictions.csv')

        # loss calculation is not as accurate here, calculate again from raw data alone
        data = pd.DataFrame({
            'mse_loss_scaled': total_loss,
            'mae_loss': total_mae_loss
        })
        data.to_csv(f'results/data/MOMENT_Demand_pl{args.pred_len}_{stage}_losses.csv')

    avg_loss = np.average(total_loss)
    avg_mae_loss = np.average(total_mae_loss)

    avg_nems_forecast_loss = np.average(total_forecast_loss)
    avg_nems_forecast_mae_loss = np.average(total_forecast_mae_loss)

    return avg_loss, avg_mae_loss, avg_nems_forecast_loss, avg_nems_forecast_mae_loss


def create_datetime(row):
    row = row.astype(int)
    return datetime(row['year'], row['month'], row['day'], row['hour'], row['minute'])


# Forecasting task
print(f'[DEBUG]: Forecasting for horizon length {args.pred_len}...')

# if is_zero_shot_short_horizon:
#     model = MOMENTPipeline.from_pretrained(
#         "AutonLab/MOMENT-1-large",
#         model_kwargs={
#             'task_name': 'short-horizon-forecasting',
#             'forecast_horizon': args.pred_len
#         },
#     )
# else:
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        'task_name': 'long-horizon-forecasting',
        'forecast_horizon': args.pred_len
    },
)
model.init()
if torch.cuda.device_count() > 1:
    print("Detected", torch.cuda.device_count(), "GPUs. Initialising Multi-GPU configuration...")
    model = DataParallel(model)
model.to(device)

train_data, train_loader = data_provider(args, 'train')
vali_data, vali_loader = data_provider(args, 'val')
test_data, test_loader = data_provider(args, 'test')

# Pre-evaluation before training
model.eval()
with torch.no_grad():
    # validation
    vali_loss, vali_mae_loss, vali_forecast_loss, vali_forecast_mae_loss = val_or_test(vali_loader)
    # test
    test_loss, test_mae_loss, test_forecast_loss, test_forecast_mae_loss = val_or_test(test_loader, True, 'base')
model.train()

print(
    "Horizon: {0} Pre-eval | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
        args.pred_len, vali_loss, vali_mae_loss, test_loss, test_mae_loss))
print(
    "Horizon: {0} NEMS forecasted | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
        args.pred_len, vali_forecast_loss, vali_forecast_mae_loss, test_forecast_loss, test_forecast_mae_loss))

# if is_zero_shot_short_horizon:
#     print('Zero shot short horizon forecasting, ending inference...')
#     exit()

# ---- Training ----
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Create a OneCycleLR scheduler
max_lr = args.learning_rate
total_steps = len(train_loader)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=args.pct_start)

# Gradient clipping value
max_norm = 5.0

losses = []
start = time.time()
for (batch_x, batch_y, batch_x_mark, batch_y_mark,
     nems_forecast_x, nems_forecast_y, input_mask) in tqdm(train_loader, total=len(train_loader)):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    input_mask = input_mask.float().to(device)

    optimizer.zero_grad(set_to_none=True)

    if args.use_amp:
        with torch.cuda.amp.autocast():
            output = model(batch_x, input_mask)

            train_loss = criterion(output[:, target_index, :], batch_y[:, target_index, :])
            # train_loss = criterion(torch.from_numpy(output), batch_y)

            # Scales the loss for mixed precision training
            scaler.scale(train_loss).backward()

            # Clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
    else:
        output = model(batch_x, input_mask)

        train_loss = criterion(output[:, target_index, :], batch_y[:, target_index, :])

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    losses.append(train_loss.item())

elapsed = time.time() - start

losses = np.array(losses)
average_loss = np.average(losses)
print(f"Epoch 1: Train loss: {average_loss:.3f}\n")
print(f'Time elapsed: {elapsed}')

# step the learning rate scheduler
scheduler.step()

# ---- Evaluation ----
model.eval()
with torch.no_grad():
    # validation
    vali_loss, vali_mae_loss, vali_forecast_loss, vali_forecast_mae_loss = val_or_test(vali_loader)
    # test
    test_loss, test_mae_loss, test_forecast_loss, test_forecast_mae_loss = val_or_test(test_loader, True, 'post-lp')
model.train()

print(
    "Horizon: {0} LP-eval | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
        args.pred_len, vali_loss, vali_mae_loss, test_loss, test_mae_loss))
print(
    "Horizon: {0} NEMS forecasted | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}\n".format(
        args.pred_len, vali_forecast_loss, vali_forecast_mae_loss, test_forecast_loss, test_forecast_mae_loss))

torch.save(model.state_dict(), str(path) + '/' + 'checkpoint')