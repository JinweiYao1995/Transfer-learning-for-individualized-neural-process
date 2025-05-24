# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:01:38 2024

@author: yaojinwei
"""

import torch 


class TrainCollator:
    def __init__(self, target_size, context_size, tasks):
        self.target_size = target_size
        self.context_size = context_size
        self.tasks = tasks
        
    def __call__(self, batch):
        Xperm, Yperm, X_D, Y_D, X_comp, Y_comp = zip(*batch) #Transposes a list of tuples, effectively grouping all elements at each position
        X_D_stack = {task: torch.stack([X_D_i[task] for X_D_i in X_D]) for task in self.tasks}
        Y_D_stack = {task: torch.stack([Y_D_i[task] for Y_D_i in Y_D]) for task in self.tasks}
        X_comp = X_comp[0]
        Y_comp_stack = {task:torch.stack([Y_comp_i[task] for Y_comp_i in Y_comp]) for task in self.tasks}
                        
        
        X_C = {task: torch.stack([X_P_i[task][0:self.context_size[task]] for X_P_i in Xperm]) for task in self.tasks}
        Y_C = {task: torch.stack([Y_P_i[task][0:self.context_size[task]] for Y_P_i in Yperm]) for task in self.tasks}
                  
        return X_C, Y_C, X_D_stack, Y_D_stack, X_comp, Y_comp_stack


def to_device(data, device):
    '''
    Load data with arbitrary structure on device.
    '''
    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
            
    return to_device_wrapper(data)


def get_data_iterator(data_loader, device):
    '''
    Iterator wrapper for dataloader
    '''
    def get_batch():
        while True:
            for batch in data_loader:
                yield to_device(batch, device)
    return get_batch()