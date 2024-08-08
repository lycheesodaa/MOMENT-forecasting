import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from torch.nn.functional import mse_loss
from tqdm import tqdm
from utils.tools import MAPELoss

save_path = 'figures/demand/'

predlens = [96, 192, 356]
# predlens = [1, 12, 72]
types = ['base', 'post-lp']

# pred_len=predlens[0]
# d_model=dms[0]

for pred_len in tqdm(predlens):
    dfs = []
    for train_type in tqdm(types):
        # load the data and halve it (data processing accidentally concatenated two copies of the same data)
        dfs.append(pd.read_csv(f'results/data/MOMENT_Demand_pl{pred_len}_{train_type}_predictions.csv', index_col=0))
    dfs[0]['post_lp_pred'] = dfs[1]['pred']
    df = dfs[0].rename(columns={
        'pred': 'base_pred'
    })

    midpoint = len(df) // 2
    if df.iloc[:midpoint].equals(df.iloc[midpoint:].reset_index(drop=True)):
        df = df.iloc[:midpoint]
    df['date'] = pd.to_datetime(df['date'])

    # re-insert the forecast column since the data processing might have meddled with it
    raw_data = pd.read_csv('data/demand_data_all_cleaned.csv')
    raw_data['date'] = pd.to_datetime(raw_data['datetime'])
    raw_data = raw_data[['date', 'forecast']]
    df.drop(columns=['nems_forecast'], inplace=True)
    df = pd.merge(df, raw_data, on='date', how='left')
    if df.isna().any().any():
        exit()

    if pred_len >= 12:
        # single iter of length pred_len (second pred_len horizon)
        single = df[pred_len ** 2:pred_len ** 2 + pred_len]
        min_date = single['date'].min()
        max_date = single['date'].max()
        single = single.melt(id_vars='date', value_vars=['post_lp_pred', 'true', 'forecast'], var_name='ts', value_name='demand')
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(data=single, x='date', y='demand', hue='ts')
        plt.title(f'Demand Plot (Single iteration, Horizon {pred_len})')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend(title='')
        fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_predictions_1-iter.png', bbox_inches='tight')
        plt.show()
        # exit()

        # multiple iters of the second pred_len horizon
        multiple = df.query(f"'{min_date}' <= date <= '{max_date}'")
        multiple = multiple.melt(id_vars='date', value_vars=['post_lp_pred', 'true', 'forecast'], var_name='ts', value_name='demand')
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(data=multiple, x='date', y='demand', hue='ts', errorbar='pi') # use 'sd' or 'pi'
        plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend(title='')
        fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_predictions_all-iter_pi.png', bbox_inches='tight')
        plt.show()
        # exit()

        # split-iters
        df['window'] = df.index // pred_len
        split = df.query(f"'{min_date}' <= date <= '{max_date}'")
        split = split.melt(id_vars=['date', 'window'], value_vars=['post_lp_pred', 'true', 'forecast'], var_name='ts', value_name='demand')
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(data=split, x='date', y='demand', hue='ts', style='window', linewidth=0.5, dashes=False)
        plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
        plt.xlabel('Date')
        plt.ylabel('Demand')

        custom_lines = [Line2D([0], [0], color='tab:blue', lw=1),
                        Line2D([0], [0], color='tab:orange', lw=1),
                        Line2D([0], [0], color='tab:green', lw=1)]

        plt.legend(custom_lines, ['post_lp_pred', 'true', 'forecast'])
        fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_predictions_split.png', bbox_inches='tight')
        plt.show()
    else:
        # split-iters
        df['window'] = df.index // pred_len
        min_date = df.iloc[12]['date']
        max_date = df.iloc[12 * 2]['date']
        split = df.query(f"'{min_date}' <= date <= '{max_date}'")
        split = split.melt(id_vars=['date', 'window'], value_vars=['post_lp_pred', 'true', 'forecast'], var_name='ts', value_name='demand')
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(data=split, x='date', y='demand', hue='ts', style='window', dashes=False, markers=True)
        plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend(title='')
        fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_predictions_split.png', bbox_inches='tight')
        plt.show()

    # calculating losses - MSE and MAPE
    # MSE loss is 1/n * sum((y - ypred)^2)
    # MAPE is 100%/n * sum(abs((y - ypred) / y))
    batch_len = pred_len * 16
    if len(df) % batch_len != 0:
        raise Exception(f'Error with number of iterations')
    num_iters = len(df) // batch_len

    mse_losses = []
    mape_losses = []
    lp_mse_losses = []
    lp_mape_losses = []
    forecast_mse_losses = []
    forecast_mape_losses = []
    for itr in range(num_iters):
        data_range = df[batch_len * itr:batch_len * itr + batch_len]
        pred = torch.tensor(data_range['base_pred'].values)
        lp_pred = torch.tensor(data_range['post_lp_pred'].values)
        forecast = torch.tensor(data_range['forecast'].values)
        true = torch.tensor(data_range['true'].values)

        mse = mse_loss(pred, true).item()
        mape = MAPELoss(pred, true).item()
        lp_mse = mse_loss(lp_pred, true).item()
        lp_mape = MAPELoss(lp_pred, true).item()
        forecast_mse = mse_loss(forecast, true).item()
        forecast_mape = MAPELoss(forecast, true).item()

        mse_losses.append(mse)
        mape_losses.append(mape)
        lp_mse_losses.append(lp_mse)
        lp_mape_losses.append(lp_mape)
        forecast_mse_losses.append(forecast_mse)
        forecast_mape_losses.append(forecast_mape)

    df_losses = pd.DataFrame({
        'base_mse': mse_losses,
        'base_mape': mape_losses,
        'lp_mse': lp_mse_losses,
        'lp_mape': lp_mape_losses,
        'forecast_mse': forecast_mse_losses,
        'forecast_mape': forecast_mape_losses
    })

    # df_losses = pd.read_csv(f'results/data/MOMENT_Demand_pl96_{train_type_losses.csv')
    fig = plt.figure(figsize=(12,8))
    sns.lineplot(data=df_losses, x=df_losses.index, y='lp_mse', label='lp_mse_loss', color='blue')
    sns.lineplot(data=df_losses, x=df_losses.index, y='forecast_mse', label='forecast_mse_loss', color='orange')
    plt.title('Loss Plot')
    plt.xlabel('Batch')
    plt.ylabel('MSE Loss')
    plt.legend(title='Time Series')
    fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_loss_mse.png', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(12,8))
    sns.lineplot(data=df_losses, x=df_losses.index, y='lp_mape', label='lp_mape_loss', color='blue')
    sns.lineplot(data=df_losses, x=df_losses.index, y='forecast_mape', label='forecast_mape_loss', color='orange')
    plt.title('Loss Plot')
    plt.xlabel('Batch')
    plt.ylabel('Percentage Loss')
    plt.legend(title='Time Series')
    fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_loss_mape.png', bbox_inches='tight')
    plt.show()