import calendar
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
cols_to_process = ['moment', 'moment_lp', 'moirai_ft', 'lag_llama_ft', 'lstm']

plot_lianlian_tasks = True
compare_lp_vs_base_loss = False


def halve_if_duplicated(data):
    # halve the data (data processing accidentally concatenated two copies of the same data)
    midpoint = len(data) // 2
    if data.iloc[:midpoint].equals(data.iloc[midpoint:].reset_index(drop=True)):
        data = data.iloc[:midpoint]
    data['date'] = pd.to_datetime(data['date'])

    return data


for pred_len in tqdm(predlens):
    # load MOMENT predictions
    df = pd.read_csv(f'results/data/MOMENT_Demand_pl{pred_len}_base_predictions.csv', index_col=0)
    if pred_len > 12 or not using_short_horizon_forecasting:
        post_lp = pd.read_csv(f'results/data/MOMENT_Demand_pl{pred_len}_post-lp_predictions.csv', index_col=0)
        df['moment_lp'] = post_lp['pred']
    df.rename(columns={
        'pred': 'moment'
    }, inplace=True)
    df = halve_if_duplicated(df)

    # load LSTM predictions into df
    lstm = pd.read_csv(f'results/data/LSTM_Demand_pl{pred_len}_dm200_predictions.csv', index_col=0)
    lstm = halve_if_duplicated(lstm)
    df['lstm'] = lstm['pred']

    # load MOIRAI predictions into df
    moirai = pd.read_csv(f'results/data/MOIRAI_pl{pred_len}_zero_shot.csv')
    moirai = halve_if_duplicated(moirai)
    df['moirai_zs_mean'] = moirai['pred_mean']
    df['moirai_zs'] = moirai['pred_median']
    moirai = pd.read_csv(f'results/data/MOIRAI_pl{pred_len}_finetuned.csv')
    moirai = halve_if_duplicated(moirai)
    df['moirai_ft_mean'] = moirai['pred_mean']
    df['moirai_ft'] = moirai['pred_median']

    # load Lag-Llama predictions into df
    lag_llama = pd.read_csv(f'results/data/MOIRAI_pl{pred_len}_finetuned.csv')
    lag_llama = halve_if_duplicated(lag_llama)
    df['lag_llama_ft_mean'] = lag_llama['pred_mean']
    df['lag_llama_ft'] = lag_llama['pred_median']

    # re-insert the forecast column since the data processing might have meddled with it
    raw_data = pd.read_csv('data/demand_data_all_cleaned.csv')
    raw_data['date'] = pd.to_datetime(raw_data['datetime'])
    raw_data_forecast = raw_data[['date', 'forecast']]
    df.drop(columns=['nems_forecast'], inplace=True)
    df = pd.merge(df, raw_data_forecast, on='date', how='left')
    if df.isna().any().any():
        exit()

    df.set_index('date', inplace=True)

    data_dict = {}
    for col in cols_to_process:
        data_dict[col] = {}

    # get losses from each day
    day_dfs = {}
    for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
        day_dfs[day] = df[df.index.dayofweek == i]

    for day, day_df in day_dfs.items():
        for col in cols_to_process:
            mape = calculate_mape(day_df[col].values, day_df['true'].values)
            data_dict[col][day] = mape

    # get losses from each month
    month_dfs = {}
    for i in range(1,13):
        month_dfs[i] = df[df.index.month == i]

    for month, month_df in month_dfs.items():
        for col in cols_to_process:
            mape = calculate_mape(month_df[col].values, month_df['true'].values)
            data_dict[col][calendar.month_name[month]] = mape

    # get losses from each hour?
    hour_dfs = {}
    for i in range(0,24):
        hour_dfs[i] = df[df.index.hour == i]

    for hour, hour_df in hour_dfs.items():
        for col in cols_to_process:
            mape = calculate_mape(hour_df[col].values, hour_df['true'].values)
            data_dict[col]['hour_' + str(hour)] = mape

    pd.DataFrame(data_dict).to_csv(save_path + f'period_based_losses_pl{pred_len}.csv')

