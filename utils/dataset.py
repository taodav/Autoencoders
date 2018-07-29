import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_loaders(dataset, valid_size=0.1, shuffle=True, seed=1234, batch_size=32):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

