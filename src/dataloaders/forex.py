import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,Subset
from src.dataloaders.base import default_data_path, SequenceDataset

class ForexDataset(Dataset):
    def __init__(self, csv_file, data_dir, seq_len=30, pred_len = 1):
        self.seq_len = 30
        self.pred_len = 1
        self.data = pd.read_csv(os.path.join(data_dir,csv_file), header = None)
        self.data.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'amount']
        self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
        self.data_set_index('datetime', inplace = True)
        self.data.drop(['date', 'time', 'high', 'low', 'close', 'amount'], axis=1, inplace=True)
        self.data = self.data[['open']]
        

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        data = self.data['open'].values[idx:idx + self.seq_len]
        target = self.data['open'].values[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return np.array(data), np.array(target)
        

class ForexSequenceDataset(SequenceDataset):
    _name_ = 'forex'
    d_input = 1
    d_output = 1
    l_output = 0
    
    @property
    def init_defaults(self):
        return {
            "val_split": 0.1,
            "seed": 42,
            "data_dir": "./data/informer/forex",
            "seq_len": 30,
            "pred_len": 1
        }

    def setup(self):
        csv_files = [file for file in os.listdir(self.data_dir) if file.endswith(".csv")]
        full_dataset = ForexDataset(csv_files[0], self.data_dir, seq_len = self.seq_len, pred_len = self.pred_len)
        total_len = len(full_dataset)
        train_len = int(total_len * (1.0 - self.val_split))
        val_len = total_len - train_len

        indices = list(range(total_len))
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        self.dataset_train = Subset(full_dataset, train_indices)
        self.dataset_val = Subset(full_dataset, val_indices)
        
    def init(self):
        pass

    def _dataloader(self, dataset,  **kwargs):
        return DataLoader(dataset, batch_size=kwargs.get('batch_size', 32), num_workers=kwargs.get('num_workers', 0), **kwargs)