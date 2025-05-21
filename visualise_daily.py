import os
from datetime import timedelta

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.tools import calculate_mse, calculate_mape

save_path = 'figures/demand/daily/'
data_dir = 'results/data/daily'

predlens = [1, 3, 7, 14, 30, 60, 178, 356]

batch_size = 16
using_short_horizon_forecasting = False
plot_lianlian_tasks = False
compare_lp_vs_base_loss = False
value_vars_list = ['moment_zs', 'moment',
                   'moirai_zs', 'moirai',
                   # 'lag_llama', 'lag_llama_zs',
                   'ttm_zs', 'ttm',
                   # 'chronos_zs', 'chronos',
                   'lstm', 'conv_lstm', 'gru', 'gru_att',
                   # 'arima',
                   'true', 'forecast']
vars_name_map = {
    'moment': 'MOMENT',
    'moment_zs': 'MOMENT_zeroshot',
    'moirai': 'MOIRAI',
    'moirai_zs': 'MOIRAI_zeroshot',
    'lag_llama': 'Lag-Llama',
    'lag_llama_zs': 'Lag-Llama_zeroshot',
    'ttm': 'TTMs',
    'ttm_zs': 'TTMs_zeroshot',
    # 'ttm_5shot': 'TTMs_5shot',
    'chronos_zs': 'Chronos_zeroshot',
    'chronos': 'Chronos',
    'lstm': 'LSTM',
    'conv_lstm': 'Conv-LSTM',
    'gru': 'GRU',
    'gru_att': 'GRU-Attention',
    'forecast': 'NEM Forecast'
}


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
    to_exclude = []

    # load MOMENT predictions
    df = pd.read_csv(data_dir + f'/MOMENT_Demand_pl{_pred_len}_base_predictions.csv', index_col=0)
    if _pred_len > 12 or not using_short_horizon_forecasting:
        post_lp = pd.read_csv(data_dir + f'/MOMENT_Demand_pl{_pred_len}_post-lp_predictions.csv', index_col=0)
        df['moment'] = post_lp['pred']
    df.rename(columns={
        'pred': 'moment_zs'
    }, inplace=True)
    df = halve_if_duplicated(df)

    try:
        # load MOIRAI predictions into df
        moirai = pd.read_csv(data_dir + f'/MOIRAI_sg_daily_pl{_pred_len}_zero_shot.csv')
        moirai = halve_if_duplicated(moirai)
        df['moirai_zs'] = moirai['pred_mean']
        moirai = pd.read_csv(data_dir + f'/MOIRAI_sg_daily_pl{_pred_len}_finetuned.csv')
        moirai = halve_if_duplicated(moirai)
        df['moirai'] = moirai['pred_mean']
    except FileNotFoundError as e:
        print(e)
        to_exclude.append('moirai')
        to_exclude.append('moirai_zs')

    # try:
    #     # load Lag-Llama predictions into df
    #     # accidentally removed SG Lag-Llama predictions, but since they have already been calculated, its okay
    #     lag_llama = pd.read_csv(data_dir + f'/Lag-Llama_pl{_pred_len}_zero_shot.csv')
    #     lag_llama = halve_if_duplicated(lag_llama)
    #     df['lag_llama_zs'] = lag_llama['pred_mean']
    #     lag_llama = pd.read_csv(data_dir + f'/Lag-Llama_pl{_pred_len}_finetuned.csv')
    #     lag_llama = halve_if_duplicated(lag_llama)
    #     df['lag_llama'] = lag_llama['pred_mean']
    # except FileNotFoundError as e:
    #     print(e)
    #     to_exclude.append('lag_llama')
    #     to_exclude.append('lag_llama_zs')

    # load LSTM predictions into df
    lstm = pd.read_csv(data_dir + f'/LSTM_Demand_pl{_pred_len}_dm200_predictions.csv', index_col=0)
    lstm = halve_if_duplicated(lstm)
    df['lstm'] = lstm['pred']

    # load ConvLSTM predictions into df
    conv_lstm = pd.read_csv(data_dir + f'/ConvLSTM_Demand_pl{_pred_len}_dm200_predictions.csv')
    conv_lstm = halve_if_duplicated(conv_lstm)
    df['conv_lstm'] = conv_lstm['pred']

    # load GRU predictions into df
    gru = pd.read_csv(data_dir + f'/GRU_Demand_pl{_pred_len}_dm200_predictions.csv')
    gru = halve_if_duplicated(gru)
    df['gru'] = gru['pred']

    # load GRU_Attention predictions into df
    gru_att = pd.read_csv(data_dir + f'/GRUAttention_Demand_pl{_pred_len}_dm200_predictions.csv')
    gru_att = halve_if_duplicated(gru_att)
    df['gru_att'] = gru_att['pred']

    # # load ARIMA predictions into df
    # arima = pd.read_csv(data_dir + f'/ARIMA_{_pred_len}_predictions.csv')
    # arima = halve_if_duplicated(arima)
    # df['arima'] = arima['pred']

    # # load Chronos predictions into df
    # if _pred_len <= 60:
    #     chronos = pd.read_csv(data_dir + f'/Chronos_pl{_pred_len}_zero_shot.csv')
    #     df['chronos_zs'] = chronos['pred']
    # chronos = pd.read_csv(data_dir + f'/Chronos_pl{_pred_len}_finetuned.csv')
    # df['chronos'] = chronos['pred']

    try:
        # load TTMs predictions into df
        ttm = pd.read_csv(data_dir + f'/TTMs_pl{_pred_len}_zeroshot.csv')
        # TTMs has the correct number of windows; not sure why the rest don't -> shorten df to match
        df = df.iloc[_pred_len:len(ttm) + _pred_len].reset_index()
        df['ttm_zs'] = ttm['actual']
        # ttm = pd.read_csv(data_dir + f'/TTMs_pl{_pred_len}_fewshot5.csv')
        # df['ttm_5shot'] = ttm['actual']
        ttm = pd.read_csv(data_dir + f'/TTMs_pl{_pred_len}_fullshot.csv')
        df['ttm'] = ttm['actual']
    except FileNotFoundError as e:
        print(e)
        to_exclude.append('ttm')
        to_exclude.append('ttm_zs')

    return df, to_exclude


def plot_multiple(df, to_exclude, title, path_to_save):
    columns_to_plot = [col for col in value_vars_list if col != 'true']
    columns_to_plot = [col for col in columns_to_plot if col not in to_exclude]

    # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 15), sharex=True, sharey=True)
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

    curr_vars = [value for value in value_vars_list if value not in to_exclude]
    if pred_len > 60:
        curr_vars = [col for col in curr_vars if col != "chronos_zs"]

    # re-insert the forecast column since the data processing might have meddled with it
    raw_data = pd.read_csv('data/demand_data_all_cleaned.csv')
    raw_data['date'] = pd.to_datetime(raw_data['datetime'])
    raw_data_forecast = raw_data[['date', 'forecast']]
    # df.drop(columns=['nems_forecast'], inplace=True)
    df = pd.merge(df, raw_data_forecast, on='date', how='left')
    if df.isna().any().any():
        # droplast if the number of missing windows < 20, otherwise there may be an error
        last_valid_index = df.dropna().index[-1]
        print(f'Dropped {(len(df) - last_valid_index) // pred_len} windows')
        df.dropna(inplace=True)

    monday_indexes = []
    mondays_temp = raw_data.iloc[-int(len(raw_data) * 0.2 + pred_len):-pred_len][(raw_data['hour'] == 0) & (raw_data['day_of_week'] == 'Monday')].sample(n=1)
    monday_indexes.extend(list(mondays_temp.index))
    # monday_indexes = [55680, 56016, 56520, 52824, 58200, 53496]
    monday_indexes = sorted(monday_indexes)
    mondays = raw_data.iloc[monday_indexes]
    for i, (index, monday) in enumerate(mondays.iterrows()):
        start_date = monday['date']
        end_date = start_date + timedelta(days=7)
        sampled_week = df.query(f"'{start_date}' <= date <= '{end_date}'")
        # print(sampled_week['date'].max())
        plot_multiple(sampled_week, to_exclude,
                      f'Week #{i + 1} (Horizon {pred_len})',
                      save_path + f'Demand_pl{pred_len}_predictions_week{i + 1}.png')

    # calculating losses
    cols_to_process = [col for col in curr_vars if col != "true"]

    if pred_len <= 12 and using_short_horizon_forecasting:
        cols_to_process = [col for col in cols_to_process if col != "moment"]

    if compare_lp_vs_base_loss:
        cols_to_process = ['moment_zs', 'moment']
        save_path = 'figures/demand/lp_vs_base_loss/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # export average predictions
    avg_data_dict = {'mse': {}, 'mape': {}}
    for col in cols_to_process:
        mse = calculate_mse(df[col].values, df['true'].values)
        mape = calculate_mape(df[col].values, df['true'].values)

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