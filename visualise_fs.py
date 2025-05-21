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

save_path = 'figures/demand/feature_comparison/'

# predlens = [96, 192, 356]
# predlens = [1, 12, 72]
predlens = [1, 12, 24, 48, 72, 168, 336]
# predlens = [1, 7, 14, 30, 60, 178, 356]

batch_size = 16
using_short_horizon_forecasting = False
value_vars_list = ['lstm', 'lstm9', 'lstm5', 'lstm0',
                   # 'moment', 'moment_zs', 'moment9', 'moment_zs9', 'moment5', 'moment_zs5', 'moment0', 'moment_zs0',
                   # 'moirai', 'moirai_zs', 'moirai9', 'moirai_zs9', 'moirai5', 'moirai_zs5', 'moirai0', 'moirai_zs0',
                   # 'ttms', 'ttms9', 'ttms5', 'ttms0',
                   # 'gru', 'gru9', 'gru5', 'gru0',
                   'true']
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

plot_lianlian_tasks = False
compare_lp_vs_base_loss = False


def halve_if_duplicated(data):
    # halve the data (data processing accidentally concatenated two copies of the same data)
    midpoint = len(data) // 2
    if data.iloc[:midpoint].equals(data.iloc[midpoint:].reset_index(drop=True)):
        data = data.iloc[:midpoint]
        print('halved')
    data['date'] = pd.to_datetime(data['date'])

    return data


'''
Add in additional dataframes here and rename pred columns to match the added name in value_vars_list  
'''
def load_data(_pred_len):
    to_exclude = []

    # load LSTM predictions into df
    df = pd.read_csv(f'results/data/feature_select_top5/LSTM_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    df.rename(columns={
        'pred': 'lstm5'
    }, inplace=True)
    lstm = pd.read_csv(f'results/data/feature_select_top9/LSTM_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    df['lstm9'] = lstm['pred']
    lstm = pd.read_csv(f'results/data/feature_select_top0/LSTM_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    df['lstm0'] = lstm['pred']
    lstm = pd.read_csv(f'results/data/LSTM_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    df['lstm'] = lstm['pred']

    # # load GRU predictions into df
    # gru = pd.read_csv(f'results/data/feature_select_top5/GRU_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    # gru = halve_if_duplicated(gru)
    # df['gru5'] = gru['pred']
    # gru = pd.read_csv(f'results/data/feature_select_top9/GRU_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    # gru = halve_if_duplicated(gru)
    # df['gru9'] = gru['pred']
    # gru = pd.read_csv(f'results/data/feature_select_top0/GRU_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    # gru = halve_if_duplicated(gru)
    # df['gru0'] = gru['pred']
    # gru = pd.read_csv(f'results/data/GRU_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    # gru = halve_if_duplicated(gru)
    # df['gru'] = gru['pred']

    # try:
    #     # load MOMENT predictions
    #     base = pd.read_csv(f'results/data/feature_select_top5/MOMENT_Demand_pl{_pred_len}_base_predictions.csv', index_col=0)
    #     df['moment_zs5'] = base['pred']
    #     post_lp = pd.read_csv(f'results/data/feature_select_top5/MOMENT_Demand_pl{_pred_len}_post-lp_predictions.csv', index_col=0)
    #     df['moment5'] = post_lp['pred']
    #     df = halve_if_duplicated(df)
    #     base = pd.read_csv(f'results/data/feature_select_top9/MOMENT_Demand_pl{_pred_len}_base_predictions.csv', index_col=0)
    #     base = halve_if_duplicated(base)
    #     df['moment_zs9'] = base['pred']
    #     post_lp = pd.read_csv(f'results/data/feature_select_top9/MOMENT_Demand_pl{_pred_len}_post-lp_predictions.csv', index_col=0)
    #     post_lp = halve_if_duplicated(post_lp)
    #     df['moment9'] = post_lp['pred']
    #     base = pd.read_csv(f'results/data/feature_select_top9/MOMENT_Demand_pl{_pred_len}_base_predictions.csv', index_col=0)
    #     base = halve_if_duplicated(base)
    #     df['moment_zs0'] = base['pred']
    #     post_lp = pd.read_csv(f'results/data/feature_select_top0/MOMENT_Demand_pl{_pred_len}_post-lp_predictions.csv', index_col=0)
    #     post_lp = halve_if_duplicated(post_lp)
    #     df['moment0'] = post_lp['pred']
    #     base = pd.read_csv(f'results/data/MOMENT_Demand_pl{_pred_len}_base_predictions.csv', index_col=0)
    #     base = halve_if_duplicated(base)
    #     df['moment_zs'] = base['pred']
    #     post_lp = pd.read_csv(f'results/data/MOMENT_Demand_pl{_pred_len}_post-lp_predictions.csv', index_col=0)
    #     post_lp = halve_if_duplicated(post_lp)
    #     df['moment'] = post_lp['pred']
    # except Exception as e:
    #     print(e)
    #
    # try:
    #     # load MOIRAI predictions
    #     moirai = pd.read_csv(f'results/data/feature_select_top5/MOIRAI_top5_pl{_pred_len}_zero_shot.csv')
    #     moirai = halve_if_duplicated(moirai)
    #     df['moirai_zs5'] = moirai['pred_mean']
    #     moirai_ft = pd.read_csv(f'results/data/feature_select_top5/MOIRAI_top5_pl{_pred_len}_finetuned.csv')
    #     moirai_ft = halve_if_duplicated(moirai_ft)
    #     df['moirai5'] = moirai_ft['pred_mean']
    #     moirai = pd.read_csv(f'results/data/feature_select_top9/MOIRAI_top9_pl{_pred_len}_zero_shot.csv')
    #     moirai = halve_if_duplicated(moirai)
    #     df['moirai_zs9'] = moirai['pred_mean']
    #     moirai_ft = pd.read_csv(f'results/data/feature_select_top9/MOIRAI_top9_pl{_pred_len}_finetuned.csv')
    #     moirai_ft = halve_if_duplicated(moirai_ft)
    #     df['moirai9'] = moirai_ft['pred_mean']
    #     moirai = pd.read_csv(f'results/data/feature_select_top0/MOIRAI_top0_pl{_pred_len}_zero_shot.csv')
    #     moirai = halve_if_duplicated(moirai)
    #     df['moirai_zs0'] = moirai['pred_mean']
    #     moirai_ft = pd.read_csv(f'results/data/feature_select_top0/MOIRAI_top0_pl{_pred_len}_finetuned.csv')
    #     moirai_ft = halve_if_duplicated(moirai_ft)
    #     df['moirai0'] = moirai_ft['pred_mean']
    #     moirai = pd.read_csv(f'results/data/MOIRAI_pl{_pred_len}_zero_shot.csv')
    #     moirai = halve_if_duplicated(moirai)
    #     df['moirai_zs'] = moirai['pred_mean']
    #     moirai_ft = pd.read_csv(f'results/data/MOIRAI_pl{_pred_len}_finetuned.csv')
    #     moirai_ft = halve_if_duplicated(moirai_ft)
    #     df['moirai'] = moirai_ft['pred_mean']
    # except Exception as e:
    #     print(e)
    #
    # try:
    #     # load TTMs into df
    #     ttms = pd.read_csv(f'results/data/feature_select_top5/TTMs_pl{_pred_len}_fullshot.csv', index_col=0)
    #     df = df.iloc[_pred_len:len(ttms) + _pred_len].reset_index()
    #     df['ttms5'] = ttms['actual']
    #     ttms = pd.read_csv(f'results/data/feature_select_top9/TTMs_pl{_pred_len}_fullshot.csv', index_col=0)
    #     df['ttms9'] = ttms['actual']
    #     ttms = pd.read_csv(f'results/data/feature_select_top0/TTMs_pl{_pred_len}_fullshot.csv', index_col=0)
    #     df['ttms0'] = ttms['actual']
    #     ttms = pd.read_csv(f'results/data/TTMs_pl{_pred_len}_fullshot.csv', index_col=0)
    #     df['ttms'] = ttms['actual']
    # except Exception as e:
    #     print(e)

    return df, to_exclude


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
    df, to_exclude = load_data(pred_len)
    curr_vars = [col for col in value_vars_list if col not in to_exclude]

    if df.isna().any().any():
        last_valid_index = df.dropna().index[-1]
        print(f'Dropped {(len(df) - last_valid_index) // pred_len} windows')
        df.dropna(inplace=True)

    if plot_lianlian_tasks:
        test = df.melt(id_vars='date', value_vars=curr_vars, var_name='ts', value_name='demand')
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

    # calculating losses
    cols_to_process = [col for col in curr_vars if col != "true"]

    # export average predictions
    avg_data_dict = {'mse': {}, 'mape': {}}
    for col in cols_to_process:
        mse = calculate_mse(df[col].values, df['true'].values)
        mape = calculate_mape(df[col].values, df['true'].values)

        if mse is None or mape is None:
            raise Exception(f'Error calculating losses for pl{pred_len}, {col}.')

        avg_data_dict['mse'][col] = mse
        avg_data_dict['mape'][col] = mape

    pd.DataFrame(avg_data_dict).to_csv(save_path + f'pl{pred_len}_average_losses_comparison.csv')

dfs = []
for pred_len in predlens:
    dfs.append(pd.read_csv(save_path + f'pl{pred_len}_average_losses_comparison.csv', index_col=0)
               .add_prefix(f'pl{pred_len}_'))

combined_losses = pd.concat(dfs, axis=1)

# Reorder the columns
mape_columns = [col for col in combined_losses.columns if 'mape' in col.lower()]
mse_columns = [col for col in combined_losses.columns if 'mse' in col.lower()]
combined_losses = combined_losses[mape_columns + mse_columns]

combined_losses.to_csv(save_path + 'losses_combined.csv')