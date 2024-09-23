import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class data_set(Dataset):

    def __init__(self, data,config=None):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):

        label = torch.tensor([item['relation'] for item in data])
        neg_labels = [torch.tensor(item['neg_labels']) for item in data]
        batch_instance = {'ids': [], 'mask': []}
        batch_instance['ids'] = torch.tensor([item['ids'] for item in data])
        batch_instance['mask'] = torch.tensor([item['mask'] for item in data])
        # = [torch.tensor(item['tokens']['input_ids']) for item in data]
        return (
            label,
            neg_labels,
            # tokens # 
            batch_instance,
        )

def get_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None):

    dataset = data_set(data, config)
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader

class llm_data_set(Dataset):

    def __init__(self, data,config=None, mmi=False):
        self.data = data
        self.config = config
        self.mmi = mmi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):

        label = torch.tensor([item['relation'] for item in data])
        neg_labels = [torch.tensor(item['neg_labels']) for item in data]
        tokens = [torch.tensor(item['tokens']) for item in data]
        id = [torch.tensor(item['id']) for item in data]
        if self.mmi:
            att_mask_0 = [torch.tensor(item['att_mask_0']) for item in data]
            return (
                label,
                neg_labels,
                tokens,
                att_mask_0,
                id
            )
        return (
            label,
            neg_labels,
            tokens,
            id
        )

def get_llm_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None, mmi=False):

    dataset = llm_data_set(data, config, mmi)
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader