import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Stocks(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS',  data_path='AAPL.csv',
                 target='close', scale=True, timeenc=0, freq='d', percent=100,
                 seasonal_patterns=None):
        if size is None:
            raise Exception('Please specify the seq_len, label_len or pred_len variables.')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw = df_raw.iloc[:, :7]

        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        total_len = len(df_raw)
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)

        border1s = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]
        border2s = [train_len, train_len + val_len, total_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        input_mask = np.ones(self.seq_len)

        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end].T
        seq_y = self.data_y[r_begin:r_end].T
        seq_x_mark = self.data_stamp[s_begin:s_end].T
        seq_y_mark = self.data_stamp[r_begin:r_end].T

        return seq_x, seq_y, seq_x_mark, seq_y_mark, input_mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Demand(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS',  data_path='demand_data_all_cleaned.csv',
                 target='actual', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size is None:
            raise Exception('Please specify the seq_len, label_len or pred_len variables.')
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = 0
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.nems_scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        self.has_nems = 'forecast' in df_raw.columns

        # encode text features to numerical
        # if df_raw.shape[1] != df_raw.select_dtypes(include=np.number).shape[1]:
        #     if 'period' in df_raw.columns: df_raw['period'] = pd.Categorical(df_raw['period']).codes
        #     if 'day_of_week' in df_raw.columns: df_raw['day_of_week'] = pd.Categorical(df_raw['day_of_week']).codes
        #     if 'is_weekend' in df_raw.columns: df_raw['is_weekend'] = pd.Categorical(df_raw['is_weekend']).codes
        #     if 'weatherDesc' in df_raw.columns: df_raw['weatherDesc'] = pd.Categorical(df_raw['weatherDesc']).codes

        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2

        total_len = len(df_raw)
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)
        test_len = int(test_ratio * total_len)

        border1s = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]
        border2s = [train_len, train_len + val_len, train_len + val_len + test_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.target_scaler.fit(train_data[self.target].values.reshape(-1, 1))
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.has_nems:
            # extract the forecast column AFTER scaling
            self.nems_forecast = data[border1:border2, 2]
            data = np.delete(data, 2, axis=1)

        df_stamp = df_raw[['datetime']][border1:border2]
        df_stamp['datetime'] = pd.to_datetime(df_stamp['datetime'])
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.datetime.apply(lambda row: row.year, 1)
            df_stamp['month'] = df_stamp.datetime.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.datetime.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.datetime.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.datetime.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.datetime.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['datetime']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['datetime'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        input_mask = np.ones(self.seq_len)

        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end].T
        seq_y = self.data_y[r_begin:r_end].T
        seq_x_mark = self.data_stamp[s_begin:s_end].T
        seq_y_mark = self.data_stamp[r_begin:r_end].T
        nem_forecast_x = self.nems_forecast[s_begin:s_end].T
        nem_forecast_y = self.nems_forecast[r_begin:r_end].T

        return seq_x, seq_y, seq_x_mark, seq_y_mark, nem_forecast_x, nem_forecast_y, input_mask

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def target_inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)

class CS702TrainDataset(Dataset):
    def __init__(self, file_name="train.npy", folder_path="./dataset", seq_len=14, candidate_len=3,
                 flag='full', percent=100):
        super().__init__()
        seed = 123
        train_ratio = 0.8
        val_ratio = 0.2

        self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        self.data = self.data[:int(len(self.data) * percent // 100)]

        if flag == 'train':
            self.data, _ = random_split(self.data, [train_ratio, val_ratio], torch.Generator().manual_seed(seed))
        elif flag == 'val':
            _, self.data = random_split(self.data, [train_ratio, val_ratio], torch.Generator().manual_seed(seed))
            
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len - 1

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        input_mask = torch.zeros(512, dtype=torch.bool)
        input_mask[-self.given_seq_len:] = True
        
        seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
        cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len - 1]  # exclude the last one
        next_point = self.data[(idx + 1) * self.seq_len - 1]

        labels = torch.tensor([3.0, 1, 0])

        return seq, cdd, next_point, labels, input_mask


class CS702TestDataset(Dataset):
    def __init__(self, file_name="public.npy", folder_path="./dataset", seq_len=13, candidate_len=3,
                 flag='train', percent=100):
        super().__init__()
        self.data = np.load(os.path.join(folder_path, file_name)).astype(np.float32)
        self.seq_len = seq_len
        self.candidate_len = candidate_len
        self.given_seq_len = seq_len - candidate_len

        # self.data = self.data[:24 * self.seq_len]

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        input_mask = torch.zeros(512, dtype=torch.bool)
        input_mask[-self.given_seq_len:] = True
        
        seq = self.data[idx * self.seq_len : idx * self.seq_len + self.given_seq_len]
        cdd = self.data[idx * self.seq_len + self.given_seq_len : (idx + 1) * self.seq_len]

        return seq, cdd, input_mask