# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:18:21 2024
@author: yaojinwei
"""

from torch.utils.data import DataLoader
from .utils import get_data_iterator, TrainCollator
from .synthetic import SyntheticTrainDataset, SyntheticValidDataset


def load_data(config, device, split='trainval', individual = None):
    '''
    Load train & valid or valid data and return the iterator & loader.
    '''
    if config.data == 'synthetic':
        TrainDataset, ValidDataset = SyntheticTrainDataset, SyntheticValidDataset
    else:
        raise NotImplementedError
    #create dataloader
    if split == 'trainval':
        if individual is None:
            train_data = TrainDataset(config.data_path, config.tasks, config.target_size, config.context_size, config.imbalance, config.sample_counts)
        else: #construct the individual samples 
            train_data = TrainDataset(config.data_path, config.tasks, config.target_size, config.context_size, config.imbalance, config.sample_counts, individual)
        train_collator = TrainCollator(config.target_size, config.context_size, config.tasks)
        train_loader = DataLoader(train_data, batch_size=config.global_batch_size,  
                              shuffle=False, pin_memory=(device.type == 'cuda'),
                              drop_last=True, num_workers=config.num_workers, collate_fn=train_collator)
        train_iterator = get_data_iterator(train_loader, device)   
        
        return train_loader, train_iterator, train_data   
    
    elif  split == 'ind_valid':
        valid_data = ValidDataset(config.data_path, config.tasks, config.target_size, config.context_size, config.imbalance, config.sample_counts, individual)
        valid_collator = TrainCollator(config.target_size, config.context_size, config.tasks)
        valid_loader = DataLoader(valid_data, batch_size=config.global_batch_size,  
                              shuffle=False, pin_memory=(device.type == 'cuda'),
                              drop_last=True, num_workers=config.num_workers, collate_fn=valid_collator)
        valid_iterator = get_data_iterator(valid_loader, device)   
        
        return valid_loader, valid_iterator, valid_data   
    else: 
        raise NotImplementedError
