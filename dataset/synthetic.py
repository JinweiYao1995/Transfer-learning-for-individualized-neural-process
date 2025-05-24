# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 22:22:16 2024
@author: yaojinwei
"""

#import os
import torch
from torch.utils.data import Dataset
import random


class SyntheticDataset(Dataset):
    def __init__(self, data_path, tasks, target_size, context_size, imbalance, sample_counts, individual = None):     
        # load data
        X, Y, Xperm, Yperm, XD, YD = torch.load(data_path)
        
        self.X = {}      
        self.Y = {}
        self.XD = {}
        self.YD = {}
        self.Xperm = Xperm
        self.Yperm = Yperm
        self.tasks = tasks
        self.target_size = target_size
        self.context_size = context_size
        
        #Define how proposed handle data inbalance
        if imbalance == 'proposed':
            for idx, task in enumerate(tasks):
                ts = self.target_size[task]
                if  task == 'target2':  
                    assert individual <= Y[task].size(0), f"Not expected these many individuals: {individual} > {Y[task].size(0)}"
                    number_max = max(sample_counts)
                    self.X[task] = X[task]
                    self.Y[task] = Y[task][individual].repeat(number_max,1,1)    
                    self.XD[task] = Xperm[task][individual,0:ts]
                    self.YD[task] = Yperm[task][individual,0:ts].repeat(number_max,1,1)  
                    self.Xperm[task] = Xperm[task][individual,0:ts].repeat(number_max,1,1)
                    self.Yperm[task]= Yperm[task][individual,0:ts].repeat(number_max,1,1)                                                   
                else: 
                    number_max = max(sample_counts)
                    self.X[task] = X[task]
                    self.Y[task] = Y[task][0:number_max]
                    self.Xperm[task] = Xperm[task][0:number_max,0:ts]
                    self.Yperm[task] = Yperm[task][0:number_max,0:ts]
                    self.XD[task] = XD[task][0:ts]
                    self.YD[task] = YD[task][0:number_max,0:ts]
       
        #Define how single ANP handle data imbalance            
        if imbalance == 'single':
            num_sample = sample_counts[-1]
            for idx,task in enumerate(tasks):
                ts = self.target_size[task]
                self.X[task] = X[task]
                self.Y[task] = Y[task][:num_sample]
                self.Xperm[task] = Xperm[task][:num_sample,0:ts]
                self.Yperm[task] = Yperm[task][:num_sample,0:ts]
                self.XD[task] = XD[task][0:ts]
                self.YD[task] = YD[task][:num_sample,0:ts]
            
        #Define how up-sampling handles data imbalance  
        if imbalance == 'RS':
            number_max = max(sample_counts) #previous tried MTNP is actually mtnp minimum
            for idx,task in enumerate(tasks):
                samples = [random.randint(0,sample_counts[idx]-1) for _ in range(number_max)]
                ts = self.target_size[task]
                self.X[task] = X[task]
                self.Y[task] = Y[task][samples]
                self.Xperm[task] = Xperm[task][samples,0:ts]
                self.Yperm[task] = Yperm[task][samples,0:ts]
                self.XD[task] = XD[task][0:ts]
                self.YD[task] = YD[task][samples,0:ts]
        
        self.n_functions = max(sample_counts)
        
        
    def __len__(self):        
        return  self.n_functions

    

class SyntheticTrainDataset(SyntheticDataset):  
    def __getitem__(self, idx):
        '''
        Returns complete target.
        '''
        X_D = self.XD
        Y_D = {task:self.YD[task][idx].clone() for task in self.tasks} 
        
        X_comp = self.X
        Y_comp = {task:self.Y[task][idx].clone() for task in self.tasks} 
        
        Xperm = {task:self.Xperm[task][idx].clone() for task in self.tasks} 
        Yperm = {task:self.Yperm[task][idx].clone() for task in self.tasks} 
        
        return Xperm, Yperm, X_D, Y_D, X_comp, Y_comp


class SyntheticValidDataset(Dataset):
    def __init__(self, data_path, tasks, target_size, context_size, imbalance, sample_counts, individual):     
        # load data
        X, Y, Xperm, Yperm, XD, YD = torch.load(data_path)
        
        self.X = {}      
        self.Y = {}
        self.XD = {}
        self.YD = {}
        self.Xperm = Xperm
        self.Yperm = Yperm
        self.tasks = tasks
        self.target_size = target_size
        self.context_size = context_size
        
        #Define how minimal imbalance handles data 
        
        if imbalance == 'proposed':
            for idx, task in enumerate(tasks):
                ts = self.target_size[task]
                if  task == 'target2':  
                    assert individual <= Y[task].size(0), f"Not expected these many individuals: {individual} > {Y[task].size(0)}"
                    number_max = max(sample_counts)
                    self.X[task] = X[task]
                    self.Y[task] = Y[task][individual].repeat(number_max,1,1)    
                    self.XD[task] = Xperm[task][individual,0:ts]
                    self.YD[task] = Yperm[task][individual,0:ts].repeat(number_max,1,1)  
                    self.Xperm[task] = Xperm[task][individual,0:ts].repeat(number_max,1,1)
                    self.Yperm[task]= Yperm[task][individual,0:ts].repeat(number_max,1,1)     
                else: 
                    number_max = max(sample_counts) #previous tried MTNP is actually mtnp minimum
                    self.X[task] = X[task]
                    self.Y[task] = Y[task][0:number_max]
                    self.Xperm[task] = Xperm[task][0:number_max,0:ts]
                    self.Yperm[task] = Yperm[task][0:number_max,0:ts]
                    self.XD[task] = XD[task][0:ts]
                    self.YD[task] = YD[task][0:number_max,0:ts]
                    
        else: 
             raise NotImplementedError
            
        self.n_functions = max(sample_counts)
        
    def __len__(self):        
        return  self.n_functions
 
    def __getitem__(self, idx):
        '''
        Returns complete target.
        '''
        X_D = self.XD
        Y_D = {task:self.YD[task][idx].clone() for task in self.tasks} 
        
        X_comp = self.X
        Y_comp = {task:self.Y[task][idx].clone() for task in self.tasks} 
        
        Xperm = {task:self.Xperm[task][idx].clone() for task in self.tasks} 
        Yperm = {task:self.Yperm[task][idx].clone() for task in self.tasks} 
        
        return Xperm, Yperm, X_D, Y_D, X_comp, Y_comp



