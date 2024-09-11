from datetime import timedelta

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
from dateutil.relativedelta import relativedelta
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from torch.nn.functional import mse_loss
from tqdm import tqdm

from data_provider.data_factory import data_dict

from utils.tools import MAPELoss, calculate_mse, calculate_mape

save_path = 'figures/demand/model_comparison/'

# predlens = [96, 192, 356]
predlens = [1, 12, 72]

batch_size = 16
using_short_horizon_forecasting = False
value_vars_list = ['moment', 'moirai', 'lag_llama', 'lstm', 'conv_lstm', 'gru', 'gru_att', 'true', 'forecast']
vars_name_map = {
    'moment': 'MOMENT',
    'moirai': 'MOIRAI',
    'lag_llama': 'Lag-Llama',
    'lstm': 'LSTM',
    'conv_lstm': 'Conv-LSTM',
    'gru': 'GRU',
    'gru_att': 'GRU-Attention',
    'forecast': 'NEM Forecast'
}

plot_lianlian_tasks = True
compare_lp_vs_base_loss = False


def halve_if_duplicated(data):
    # halve the data (data processing accidentally concatenated two copies of the same data)
    midpoint = len(data) // 2
    if data.iloc[:midpoint].equals(data.iloc[midpoint:].reset_index(drop=True)):
        data = data.iloc[:midpoint]
    data['date'] = pd.to_datetime(data['date'])

    return data


'''
Add in additional dataframes here and rename pred columns to match the added name in value_vars_list  
'''
def load_data(_pred_len):
    # load MOMENT predictions
    df = pd.read_csv(f'results/data/MOMENT_Demand_pl{_pred_len}_base_predictions.csv', index_col=0)
    if _pred_len > 12 or not using_short_horizon_forecasting:
        post_lp = pd.read_csv(f'results/data/MOMENT_Demand_pl{_pred_len}_post-lp_predictions.csv', index_col=0)
        df['moment'] = post_lp['pred']
    df.rename(columns={
        'pred': 'moment_zs'
    }, inplace=True)
    df = halve_if_duplicated(df)

    # load LSTM predictions into df
    lstm = pd.read_csv(f'results/data/LSTM_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    lstm = halve_if_duplicated(lstm)
    df['lstm'] = lstm['pred']

    # load MOIRAI predictions into df
    moirai = pd.read_csv(f'results/data/MOIRAI_pl{_pred_len}_zero_shot.csv')
    moirai = halve_if_duplicated(moirai)
    df['moirai_zs_mean'] = moirai['pred_mean']
    df['moirai_zs'] = moirai['pred_median']
    moirai = pd.read_csv(f'results/data/MOIRAI_pl{_pred_len}_finetuned.csv')
    moirai = halve_if_duplicated(moirai)
    df['moirai_mean'] = moirai['pred_mean']
    df['moirai'] = moirai['pred_median']

    # load Lag-Llama predictions into df
    lag_llama = pd.read_csv(f'results/data/Lag-Llama_pl{_pred_len}_finetuned.csv')
    lag_llama = halve_if_duplicated(lag_llama)
    df['lag_llama_mean'] = lag_llama['pred_mean']
    df['lag_llama'] = lag_llama['pred_median']

    # load ConvLSTM predictions into df
    conv_lstm = pd.read_csv(f'results/data/ConvLSTM_Demand_pl{_pred_len}_dm200_predictions.csv')
    conv_lstm = halve_if_duplicated(conv_lstm)
    df['conv_lstm'] = conv_lstm['pred']

    # load GRU predictions into df
    gru = pd.read_csv(f'results/data/GRU_Demand_pl{_pred_len}_dm200_predictions.csv')
    gru = halve_if_duplicated(gru)
    df['gru'] = gru['pred']

    # load GRU_Attention predictions into df
    gru_att = pd.read_csv(f'results/data/GRUAttention_Demand_pl{_pred_len}_dm200_predictions.csv')
    gru_att = halve_if_duplicated(gru_att)
    df['gru_att'] = gru_att['pred']

    return df


def plot_multiple(df, title, path_to_save):
    columns_to_plot = [col for col in value_vars_list if col != 'true']

    # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=20)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Plot each column
    for i, col in enumerate(columns_to_plot):
        sns.lineplot(data=df, x='date', y='true', ax=axes[i], label='true')
        sns.lineplot(data=df, x='date', y=col, ax=axes[i], label=col, errorbar='pi')

        axes[i].set_title(f'{vars_name_map[col]}', fontsize=16)
        axes[i].set_xlabel('Date', fontsize=14)
        axes[i].set_ylabel('Demand', fontsize=14)
        axes[i].tick_params(axis='x', labelrotation=25)
        axes[i].legend()

    fig.savefig(path_to_save, bbox_inches='tight')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


for pred_len in tqdm(predlens):
    df = load_data(pred_len)

    # re-insert the forecast column since the data processing might have meddled with it
    raw_data = pd.read_csv('data/demand_data_all_cleaned.csv')
    raw_data['date'] = pd.to_datetime(raw_data['datetime'])
    raw_data_forecast = raw_data[['date', 'forecast']]
    df.drop(columns=['nems_forecast'], inplace=True)
    df = pd.merge(df, raw_data_forecast, on='date', how='left')
    if df.isna().any().any():
        exit()

    if plot_lianlian_tasks:
        test = df.melt(id_vars='date', value_vars=value_vars_list, var_name='ts', value_name='demand')
        fig = plt.figure(figsize=(50, 15))
        sns.lineplot(data=test, x='date', y='demand', hue='ts', errorbar='pi')
        plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})', fontsize=40)
        plt.xlabel('Date', fontsize=36)
        plt.ylabel('Demand', fontsize=36)
        plt.legend(title='', fontsize=28)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        fig.savefig(save_path + f'Demand_pl{pred_len}_predictions_all.png', bbox_inches='tight')

        # day_indexes = []
        # for i in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        #     days = raw_data.iloc[-int(len(raw_data) * 0.2 + pred_len):-pred_len][(raw_data['hour'] == 0) & (raw_data['day_of_week'] == i)].sample(n=1)
        #     day_indexes.extend(list(days.index))
        # print(day_indexes)
        day_indexes = [58200, 56544, 49848, 54072, 50904, 60168, 50616] # indexes are sampled from a previous run
        days_chosen = raw_data.iloc[day_indexes]
        for i, (index, day_) in enumerate(days_chosen.iterrows()):
            start_date = day_['date']
            end_date = start_date + timedelta(days=1)
            sampled_day = df.query(f"'{start_date}' <= date <= '{end_date}'")
            # print(sampled_day['date'].max())
            day_of_week_sampled = day_['day_of_week']
            plot_multiple(sampled_day,
                          f'Day #{i + 1} - {day_of_week_sampled} (Horizon {pred_len})',
                          save_path + f'Demand_pl{pred_len}_predictions_day{i + 1}.png')

        # indexes are sampled from a previous run
        monday_indexes = [55680, 56016, 56520, 52824, 58200, 53496]
        monday_indexes = sorted(monday_indexes)
        mondays = raw_data.iloc[monday_indexes]
        for i, (index, monday) in enumerate(mondays.iterrows()):
            start_date = monday['date']
            end_date = start_date + timedelta(days=7)
            sampled_week = df.query(f"'{start_date}' <= date <= '{end_date}'")
            # print(sampled_week['date'].max())
            plot_multiple(sampled_week,
                          f'Week #{i + 1} (Horizon {pred_len})',
                          save_path + f'Demand_pl{pred_len}_predictions_week{i + 1}.png')

        month_indexes = [54768, 58440, 59904]
        month_indexes = sorted(month_indexes)
        months = raw_data.iloc[month_indexes]
        for i, (index, monthday) in enumerate(months.iterrows()):
            start_date = monthday['date']
            end_date = start_date + relativedelta(months=1)
            sampled_month = df.query(f"'{start_date}' <= date <= '{end_date}'")
            # print(sampled_month['date'].max())
            plot_multiple(sampled_month,
                          f'Month #{i + 1} (Horizon {pred_len})',
                          save_path + f'Demand_pl{pred_len}_predictions_month{i + 1}.png')

        # continue

    # if pred_len > 1:
    #     # single iter of length pred_len (second pred_len horizon)
    #     single = df[pred_len ** 2:pred_len ** 2 + pred_len]
    #     min_date = single['date'].min()
    #     max_date = single['date'].max()
    #
    #     # multiple iters of the second pred_len horizon
    #     multiple = df.query(f"'{min_date}' <= date <= '{max_date}'")
    #     multiple = multiple.melt(id_vars='date', value_vars=value_vars_list, var_name='ts', value_name='demand')
    #     fig = plt.figure(figsize=(12,8))
    #     sns.lineplot(data=multiple, x='date', y='demand', hue='ts', errorbar='pi') # use 'sd' or 'pi'
    #     plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
    #     plt.xlabel('Date')
    #     plt.ylabel('Demand')
    #     plt.legend(title='')
    #     fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_predictions_all-iter_pi.png', bbox_inches='tight')
    #     plt.show()
    # else:
    #     # replace 'moment_lp_pred' with 'moment_pred' if using short-horizon-forecasting
    #     if using_short_horizon_forecasting:
    #         value_vars_list = value_vars_list[1:]
    #         value_vars_list.insert(0, 'moment_pred')
    #
    #     # when pred_len <= 12, use split-iters with a fixed window size of 12
    #     df['window'] = df.index // pred_len
    #     min_date = df.iloc[12]['date']
    #     max_date = df.iloc[12 * 2 - 1]['date']
    #     split = df.query(f"'{min_date}' <= date <= '{max_date}'")
    #     split = split.melt(id_vars=['date', 'window'], value_vars=value_vars_list, var_name='ts', value_name='demand')
    #     fig = plt.figure(figsize=(12,8))
    #     sns.lineplot(data=split, x='date', y='demand', hue='ts', markers=True, dashes=False)
    #     plt.title(f'Demand Plot (Multiple iterations, Horizon {pred_len})')
    #     plt.xlabel('Date')
    #     plt.ylabel('Demand')
    #
    #     cmap = get_cmap('tab10')
    #     custom_lines = []
    #     for i, val in enumerate(value_vars_list):
    #         custom_lines.append(Line2D([0], [0], color=cmap(i), lw=1))
    #
    #     plt.legend(custom_lines, value_vars_list)
    #     fig.savefig(save_path + f'MOMENT_Demand_pl{pred_len}_predictions_split.png', bbox_inches='tight')
    #     plt.show()

    # calculating losses per batch - MSE and MAPE
    batch_len = pred_len * batch_size # total batch length is pred_len * batch_size
    if len(df) % batch_len != 0:
        raise Exception(f'Error with number of iterations')
    num_iters = len(df) // batch_len

    cols_to_process = [col for col in value_vars_list if col != "true"]

    if pred_len <= 12 and using_short_horizon_forecasting:
        cols_to_process = [col for col in cols_to_process if col != "moment"]

    if compare_lp_vs_base_loss:
        cols_to_process = ['moment_zs', 'moment']
        save_path = 'figures/demand/lp_vs_base_loss/'

    data_dict = {}
    for col in cols_to_process:
        data_dict[col + '_mse'] = []
        data_dict[col + '_mape'] = []

    for itr in range(num_iters):
        data_range = df[batch_len * itr:batch_len * itr + batch_len]
        for col in cols_to_process:
            mse = calculate_mse(data_range[col].values, data_range['true'].values)
            mape = calculate_mape(data_range[col].values, data_range['true'].values)
            data_dict[col + '_mse'].append(mse)
            data_dict[col + '_mape'].append(mape)

    df_losses = pd.DataFrame(data_dict)

    # plot losses
    # mse
    fig = plt.figure(figsize=(12,8))
    for col in cols_to_process:
        col = col + '_mse'
        sns.lineplot(data=df_losses, x=df_losses.index, y=col, label=col)
    plt.title('Loss Plot')
    plt.xlabel('Batch')
    plt.ylabel('MSE Loss')
    plt.legend(title='Time Series')
    fig.savefig(save_path + f'Demand_pl{pred_len}_loss_mse.png', bbox_inches='tight')
    plt.show()

    # mape
    fig = plt.figure(figsize=(12,8))
    for col in cols_to_process:
        col = col + '_mape'
        sns.lineplot(data=df_losses, x=df_losses.index, y=col, label=col)
    plt.title('Loss Plot')
    plt.xlabel('Batch')
    plt.ylabel('MAPE Loss')
    plt.legend(title='Time Series')
    fig.savefig(save_path + f'Demand_pl{pred_len}_loss_mape.png', bbox_inches='tight')
    plt.show()

    # export average predictions
    avg_data_dict = {'mse': {}, 'mape': {}}
    for col, losses in data_dict.items():
        if col.endswith('mse'):
            avg_data_dict['mse'][col.removesuffix('_mse')] = np.mean(losses)
        elif col.endswith('mape'):
            avg_data_dict['mape'][col.removesuffix('_mape')] = np.mean(losses)
    pd.DataFrame(avg_data_dict).to_csv(save_path + f'pl{pred_len}_average_losses_comparison.csv')

dfs = []
for pred_len in predlens:
    dfs.append(pd.read_csv(save_path + f'pl{pred_len}_average_losses_comparison.csv', index_col=0))

dfs[0] = dfs[0].add_prefix('pl1_')
dfs[1] = dfs[1].add_prefix('pl12_')
dfs[2] = dfs[2].add_prefix('pl72_')

combined_losses = pd.concat(dfs, axis=1)

# Reorder the columns
mape_columns = [col for col in combined_losses.columns if 'mape' in col.lower()]
mse_columns = [col for col in combined_losses.columns if 'mse' in col.lower()]
combined_losses = combined_losses[mape_columns + mse_columns]

combined_losses.to_csv(save_path + 'losses_combined.csv')