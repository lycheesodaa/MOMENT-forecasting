import argparse
from momentfm import MOMENTPipeline
from torch.optim.lr_scheduler import OneCycleLR

from data_provider.data_factory import data_provider
from tqdm import tqdm
import random
import torch
import numpy as np
import os
import time

# horizons = [1, 2, 3, 5, 7, 14, 21, 30, 60, 96, 192, 356]
horizons = [96, 192, 356]

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
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
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
criterion = torch.nn.MSELoss().to(device)
mae_metric = torch.nn.L1Loss().to(device)


def val_or_test(loader):
    total_loss = []
    total_mae_loss = []

    for batch_x, batch_y, batch_x_mark, batch_y_mark, input_mask in tqdm(loader, total=len(loader)):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        input_mask = input_mask.to(device)


        if args.use_amp:
            with torch.cuda.amp.autocast():
                pred = model(batch_x, input_mask).forecast # this outputs an ndarray
        else:
            pred = model(batch_x, input_mask).forecast  # this outputs an ndarray

        # pred = torch.from_numpy(pred)
        true = batch_y.detach()

        loss = criterion(pred, true)
        mae_loss = mae_metric(pred, true)

        total_loss.append(loss.item())
        total_mae_loss.append(mae_loss.item())

    avg_loss = np.average(total_loss)
    avg_mae_loss = np.average(total_mae_loss)

    return avg_loss, avg_mae_loss

'''
MOMENT pipeline generally involves: 
1. pre-evaluation to test the out-of-the-box model,
2. fine-tuning on a small subset of data (also known as linear probing)
3. model evaluation post-linear probing
'''
for horizon in horizons:
    print(f'[DEBUG]: Forecasting for horizon length {horizon}...')
    args.pred_len = horizon

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': horizon
        },
    )
    model.init()
    model.to(device)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # Pre-evaluation before training
    model.eval()
    with torch.no_grad():
        # validation
        vali_loss, vali_mae_loss = val_or_test(vali_loader)
        # test
        test_loss, test_mae_loss = val_or_test(test_loader)
    model.train()

    print(
        "Horizon: {0} Pre-eval | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}\n".format(
            horizon, vali_loss, vali_mae_loss, test_loss, test_mae_loss))
    # with open('output.txt', 'a') as f:
    #     f.write("Horizon: {0} Pre-eval | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}\n".format(
    #         horizon, vali_loss, vali_mae_loss, test_loss, test_mae_loss))

    # ---- Training ----
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create a OneCycleLR scheduler
    max_lr = args.learning_rate
    total_steps = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=args.pct_start)

    # Gradient clipping value
    max_norm = 5.0

    losses = []
    for batch_x, batch_y, batch_x_mark, batch_y_mark, input_mask in tqdm(train_loader, total=len(train_loader)):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        input_mask = input_mask.float().to(device)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                output = model(batch_x, input_mask)

                train_loss = criterion(output.forecast, batch_y)
                # train_loss = criterion(torch.from_numpy(output.forecast), batch_y)

                # Scales the loss for mixed precision training
                scaler.scale(train_loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            output = model(batch_x, input_mask)

            train_loss = criterion(output.forecast, batch_y)

            train_loss.backward()
            optimizer.step()

        losses.append(train_loss.item())

    losses = np.array(losses)
    average_loss = np.average(losses)
    print(f"Epoch 1: Train loss: {average_loss:.3f}\n")

    # step the learning rate scheduler
    scheduler.step()

    # ---- Evaluation ----
    model.eval()
    with torch.no_grad():
        # validation
        vali_loss, vali_mae_loss = val_or_test(vali_loader)
        # test
        test_loss, test_mae_loss = val_or_test(test_loader)
    model.train()

    print(
        "Horizon: {0} LP-eval | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}\n".format(
            horizon, vali_loss, vali_mae_loss, test_loss, test_mae_loss))
    # with open('output.txt', 'a') as f:
    #     f.write("Horizon: {0} LP-eval | Vali Loss: {1:.7f} Vali MAE Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}\n".format(
    #         horizon, vali_loss, vali_mae_loss, test_loss, test_mae_loss))
