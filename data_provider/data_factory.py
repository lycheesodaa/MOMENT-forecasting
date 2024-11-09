from data_provider.data_loader import CS702TestDataset, CS702TrainDataset
from torch.utils.data import DataLoader

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