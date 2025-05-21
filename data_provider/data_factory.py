from data_provider.data_loader import Dataset_Stocks, Dataset_Demand, CS702TestDataset, CS702TrainDataset, \
    Dataset_Carbon, Dataset_Carbon_Monthly
from torch.utils.data import DataLoader

data_dict = {
    'Stocks': Dataset_Stocks,
    'Demand': Dataset_Demand,
    'Carbon': Dataset_Carbon,
    'Carbon_monthly': Dataset_Carbon_Monthly
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test' or 'monthly' in args.data:  # monthly data is scarce, so we don't drop last
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    if hasattr(args, 'feats_pct'):
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            scale=args.scale,
            seasonal_patterns=args.seasonal_patterns,
            feats_pct=args.feats_pct
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            scale=args.scale,
            seasonal_patterns=args.seasonal_patterns
        )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

def data_provider_cs702(args, flag):
    if flag == 'test':
        Data = CS702TestDataset
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        Data = CS702TrainDataset
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        
    data_set = Data(flag=flag, folder_path=args.root_path, percent=args.percent)
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    return data_set, data_loader