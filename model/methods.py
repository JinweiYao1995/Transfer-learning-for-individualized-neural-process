# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:40:45 2024

@author: yaojinwei
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from .Modules import SetEncoder, GlobalEncoder, TaskEncoder, ConditionalSetEncoder, MultiTaskAttention,  MultiTaskPooling,  MTPDecoder
import numpy as np

        
        
class IMTP(nn.Module):
    '''
    Implementation of the proposed method 
    '''
    def __init__(self, config):
        super().__init__()
        
        self.tasks = config.tasks
        # latent encoding path
        self.set_encoder = nn.ModuleList([SetEncoder(config.dim_x,  config.dim_ys[task],config.dim_hidden, config.module_sizes[0], config.module_sizes[1]) 
                                           for task in self.tasks]) #those are all lists of encoder
        self.global_encoder = GlobalEncoder(config.dim_hidden, config.module_sizes[2])
        
        
        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden, hierarchical=False)
                                           for task in self.tasks])

        # deterministic encoding path
        self.conditional_set_encoder = nn.ModuleList([ConditionalSetEncoder(config.dim_x,  config.dim_ys[task],config.dim_hidden, 
                                                                            config.module_sizes[0], config.module_sizes[1]) for task in self.tasks])
        self.deterministic_encoder = MultiTaskAttention(config.dim_hidden, config.module_sizes[2])
        # decoding path
        self.decoder = nn.ModuleList([MTPDecoder(config.dim_x, config.dim_ys[task], config.dim_hidden, config.module_sizes[3])
                                      for task in self.tasks])
        
        
    def state_dict_(self):
        return self.state_dict()
    
    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)           
        
    def encode_target(self, X, Y):  #Y is dict of 4 * (24,20,1)
        s = {}
        # per-task inference of latent path
        for t_idx, task in enumerate(self.tasks[:-1]):
            D_t = torch.cat((X[task], Y[task]), -1)
            s[task] = self.set_encoder[t_idx](D_t)           
        # global latent in across-task inference of latent path
        s_G = torch.stack([s[task] for task in s], 1)  #SG is (24,4,128)
        q_G = self.global_encoder(s_G)
        return q_G


    def encode_individual(self, X, Y):  #Y is dict of 4 * (24,20,1)
        s = {}
        # per-task inference of latent path
        for t_idx, task in enumerate(self.tasks[:-1]):
            D_t = torch.cat((X[task], Y[task]), -1)
            s[task] = self.set_encoder[t_idx](D_t)           
        # global latent in across-task inference of latent path
        s_I = torch.stack([s[task] for task in s], 1)  #SG is (24,4,128)
        q_I = self.global_encoder(s_I)
        return q_I

    
    def encode_task(self, X, Y):
        # task-specific latent in across-task inference of latent path
        q_T = {}
        for t_idx, task in enumerate(self.tasks[:-2]):
            D_t = torch.cat((X[task], Y[task]), -1)
            s_t = self.set_encoder[t_idx](D_t)
            q_T[task] = self.task_encoder[t_idx](s_t)            
        return q_T      
    
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        r = {} #need to vary for different task 
        # cross-attention layers in across-task inference of deterministic path
        for t_1, task_d in enumerate(self.tasks): #query for tasks 
            U_C = {}
            for t_2, task_c in enumerate(self.tasks): #context for tasks 
                C_t = torch.cat((X_C[task_c], Y_C[task_c]), -1)
                U_C[task_c] = self.conditional_set_encoder[t_2](C_t, X_C[task_c], X_D[task_d])  #now the length of U_C across task is different.
            U_C = torch.stack([U_C[task] for task in self.tasks], 1)  #now it completes one r for task 1\
            UC = self.deterministic_encoder(U_C)
            r[task_d] = UC[:,t_1]
            if task_d == 'target1':
                r[task_d] = UC[:,:-1].mean(dim = 1)
            elif task_d == 'target2':
                r[task_d] = UC.mean(dim = 1)
            else:
                r[task_d] = UC[:,t_1]
        return r 
    
    def decode(self, X, v, r):
        if not self.training:
            X = {task: X[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks}  #Check for X and r size 
            r = {task: r[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks} #the first two are batches and task
        p_Y = {}
        for t_idx, task in enumerate(self.tasks):
            p_Y[task] = self.decoder[t_idx](X[task], v[:, t_idx], r[task])
        return p_Y
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, ns_G=1, ns_T=1): #once MAP, ns_G and ns_T are irrelevant 
        if self.training:
            assert Y_D is not None
            
            q_C_G  = self.encode_target(X_C, Y_C)           
            q_D_G = self.encode_target(X_D, Y_D)
            v_G = Normal(*q_C_G).rsample() 

            q_C_I  = self.encode_individual(X_C, Y_C)           
            q_D_I = self.encode_individual(X_D, Y_D)
            v_I = Normal(*q_C_I).rsample() 

            q_C_T = self.encode_task(X_C, Y_C)    
            q_D_T = self.encode_task(X_D, Y_D)
            v_T = torch.stack([Normal(*q_C_T[task]).rsample() for task in self.tasks[:-2]], 1) 

            r = self.encode_deterministic(X_C, Y_C, X_D)  
            v = torch.cat((v_T, v_G.unsqueeze(1), v_I.unsqueeze(1),), dim=1)
            
            p_Y = self.decode(X_D, v, r)

            qct = q_C_T.copy()
            qct[self.tasks[-2]] = q_C_G
            qct[self.tasks[-1]] = q_C_I

            qdt =q_D_T.copy()
            qdt[self.tasks[-2]] = q_D_G
            qdt[self.tasks[-1]] = q_D_I
            
            #ELBO only need to measure q_C_T and q_C_T
            
            return p_Y, None, None, qdt, qct

        else:
            q_C_G = self.encode_target(X_C, Y_C)
            v_G = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)     

            q_C_I = self.encode_individual(X_C, Y_C)
            v_I = Normal(*q_C_I).sample((ns_G,)).transpose(0, 1)     

            q_C_T = self.encode_task(X_C, Y_C)
            v_T = torch.stack([Normal(*q_C_T[task]).sample((ns_T,)).transpose(0, 1)
                                 for task in q_C_T], 1)

            v = torch.cat((v_T,v_G.unsqueeze(1),v_I.unsqueeze(1)), dim=1)
            r = self.encode_deterministic(X_C, Y_C, X_D)            
            p_Y = self.decode(X_D, v, r)

            return p_Y 
        
        
        
        
class IMTPs(nn.Module):
    '''
    Implementation of the proposed method w/o sources
    '''
    def __init__(self, config):
        super().__init__()
        
        self.tasks = config.tasks
        # latent encoding path
        self.set_encoder = nn.ModuleList([SetEncoder(config.dim_x,  config.dim_ys[task],config.dim_hidden, config.module_sizes[0], config.module_sizes[1]) 
                                           for task in self.tasks]) #those are all lists of encoder
        self.global_encoder = GlobalEncoder(config.dim_hidden, config.module_sizes[2])
        
        
        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden, hierarchical=False)
                                           for task in self.tasks])

        # deterministic encoding path
        self.conditional_set_encoder = nn.ModuleList([ConditionalSetEncoder(config.dim_x,  config.dim_ys[task],config.dim_hidden, 
                                                                            config.module_sizes[0], config.module_sizes[1]) for task in self.tasks])
        self.deterministic_encoder = MultiTaskAttention(config.dim_hidden, config.module_sizes[2])
        # decoding path
        self.decoder = nn.ModuleList([MTPDecoder(config.dim_x, config.dim_ys[task], config.dim_hidden, config.module_sizes[3])
                                      for task in self.tasks])
        
        
    def state_dict_(self):
        return self.state_dict()
    
    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)           
        
    def encode_target(self, X, Y): 
        s = {}
        # per-task inference of latent path
        for t_idx, task in enumerate(self.tasks[:-1]):
            D_t = torch.cat((X[task], Y[task]), -1)
            s[task] = self.set_encoder[t_idx](D_t)           
        # global latent in across-task inference of latent path
        s_G = torch.stack([s[task] for task in s], 1)  
        q_G = self.global_encoder(s_G)
        return q_G


    def encode_individual(self, X, Y): 
        s = {}
        # per-task inference of latent path
        for t_idx, task in enumerate(self.tasks[:-1]):
            D_t = torch.cat((X[task], Y[task]), -1)
            s[task] = self.set_encoder[t_idx](D_t)           
        # global latent in across-task inference of latent path
        s_I = torch.stack([s[task] for task in s], 1)
        q_I = self.global_encoder(s_I)
        return q_I

    
    def encode_task(self, X, Y):
        # task-specific latent in across-task inference of latent path
        q_T = {}
        for t_idx, task in enumerate(self.tasks[:-2]):
            D_t = torch.cat((X[task], Y[task]), -1)
            s_t = self.set_encoder[t_idx](D_t)
            q_T[task] = self.task_encoder[t_idx](s_t)            
        return q_T      
    
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        r = {} 
        for t_1, task_d in enumerate(self.tasks): 
            U_C = {}
            for t_2, task_c in enumerate(self.tasks):
                C_t = torch.cat((X_C[task_c], Y_C[task_c]), -1)
                U_C[task_c] = self.conditional_set_encoder[t_2](C_t, X_C[task_c], X_D[task_d])  
            U_C = torch.stack([U_C[task] for task in self.tasks], 1)  
            UC = self.deterministic_encoder(U_C)
            r[task_d] = UC[:,t_1]
            if task_d == 'target1':
                r[task_d] = UC[:,:-1].mean(dim = 1)
            elif task_d == 'target2':
                r[task_d] = UC.mean(dim = 1)
            else:
                r[task_d] = UC[:,t_1]
        return r 
    
    
    def decode(self, X, v, r):
        if not self.training:
            X = {task: X[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks}  
            r = {task: r[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks} 
        p_Y = {}
        for t_idx, task in enumerate(self.tasks):
            p_Y[task] = self.decoder[t_idx](X[task], v[:, t_idx], r[task])
        return p_Y
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, ns_G=1, ns_T=1): 
        if self.training:
            assert Y_D is not None
            
            q_C_G  = self.encode_target(X_C, Y_C)          
            q_D_G = self.encode_target(X_D, Y_D)
            v_G = Normal(*q_C_G).rsample() 

            q_C_I  = self.encode_individual(X_C, Y_C)          
            q_D_I = self.encode_individual(X_D, Y_D)
            v_I = Normal(*q_C_I).rsample() 



            r = self.encode_deterministic(X_C, Y_C, X_D)  
            v = torch.cat((v_G.unsqueeze(1), v_I.unsqueeze(1),), dim=1)
            
            p_Y = self.decode(X_D, v, r)  

            qct = {}
            qct[self.tasks[-2]] = q_C_G
            qct[self.tasks[-1]] = q_C_I

            qdt = {}
            qdt[self.tasks[-2]] = q_D_G
            qdt[self.tasks[-1]] = q_D_I
            
            self.qdt = qdt
            self.qct = qct
            
            return p_Y, None, None, qdt, qct

        else:
            q_C_G = self.encode_target(X_C, Y_C)
            v_G = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1)       

            q_C_I = self.encode_individual(X_C, Y_C)
            v_I = Normal(*q_C_I).sample((ns_G,)).transpose(0, 1)          

            v = torch.cat((v_G.unsqueeze(1),v_I.unsqueeze(1)), dim=1)
            r = self.encode_deterministic(X_C, Y_C, X_D)            
            p_Y = self.decode(X_D, v, r)

            return p_Y 
        
import time as time

class MTP(nn.Module):
    '''
    Implementation of the MTNP 
    '''
    def __init__(self, config):
        super().__init__()
        
        self.tasks = config.tasks
        # latent encoding path
        self.set_encoder = nn.ModuleList([SetEncoder(config.dim_x,  config.dim_ys[task],config.dim_hidden, config.module_sizes[0], config.module_sizes[1]) 
                                           for task in self.tasks]) 
        self.global_encoder = GlobalEncoder(config.dim_hidden, config.module_sizes[2])
        
        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden)
                                           for task in self.tasks])

        # deterministic encoding path
        self.conditional_set_encoder = nn.ModuleList([ConditionalSetEncoder(config.dim_x,  config.dim_ys[task],config.dim_hidden, 
                                                                            config.module_sizes[0], config.module_sizes[1]) for task in self.tasks])
        self.deterministic_encoder = MultiTaskAttention(config.dim_hidden, config.module_sizes[2])
        # decoding path
        self.decoder = nn.ModuleList([MTPDecoder(config.dim_x, config.dim_ys[task], config.dim_hidden, config.module_sizes[3])
                                      for task in self.tasks])    
    def state_dict_(self):
        return self.state_dict()
    
    def load_state_dict_(self, state_dict):
        self.load_state_dict(state_dict)
        
        
    def encode_global(self, X, Y):
        s = {}
        # per-task inference of latent path
        for t_idx, task in enumerate(self.tasks):
            D_t = torch.cat((X[task], Y[task]), -1)
            s[task] = self.set_encoder[t_idx](D_t)           
        # global latent in across-task inference of latent path
        s_G = torch.stack([s[task] for task in s], 1)
        q_G = self.global_encoder(s_G)
        return q_G, s
    
    def encode_task(self, s, z): 
        q_T = {}
        for t_idx, task in enumerate(self.tasks):
            s_t = s[task]
            if not self.training:
                s_t = s_t.unsqueeze(1).repeat(1, z.size(1), 1)  
            q_T[task] = self.task_encoder[t_idx](s_t, z)
        return q_T
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        r = {} 
        for t_1, task_d in enumerate(self.tasks): #query for tasks 
            U_C = {}
            for t_2, task_c in enumerate(self.tasks):
                C_t = torch.cat((X_C[task_c], Y_C[task_c]), -1)
                U_C[task_c] = self.conditional_set_encoder[t_2](C_t, X_C[task_c], X_D[task_d]) 
            U_C = torch.stack([U_C[task] for task in self.tasks], 1) 
            UC = self.deterministic_encoder(U_C)
            r[task_d] = UC[:,t_1]
        return r 
    
    def decode(self, X, v, r):  
        if not self.training:
            X = {task: X[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks}  
            r = {task: r[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks} 
        p_Y = {}
        for t_idx, task in enumerate(self.tasks):
            p_Y[task] = self.decoder[t_idx](X[task], v[:, t_idx], r[task])
        return p_Y
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, ns_G=1, ns_T=1): 
        if self.training:
            assert Y_D is not None
                        
            start_time = time.time()
            q_C_G, s_C = self.encode_global(X_C, Y_C)          
            q_D_G, s_D = self.encode_global(X_D, Y_D)
            end_time = time.time()
            self.encoding_time = end_time - start_time
            
            
            start_time = time.time()
            z = Normal(*q_C_G).rsample() 
            q_C_T = self.encode_task(s_C, z)     
            q_D_T = self.encode_task(s_D, z)
            v = torch.stack([Normal(*q_C_T[task]).rsample() for task in self.tasks], 1)
            r = self.encode_deterministic(X_C, Y_C, X_D)      
            end_time = time.time()
            self.middle_time = end_time - start_time
            
            start_time = time.time()
            p_Y = self.decode(X_D, v, r)   
            end_time = time.time()
            self.decoding_time = end_time - start_time
            
            
            return p_Y, q_D_G, q_C_G, q_D_T, q_C_T
        else:  
            q_C_G, s_C = self.encode_global(X_C, Y_C)
            z = Normal(*q_C_G).sample((ns_G,)).transpose(0, 1) 
            
            q_C_T = self.encode_task(s_C, z)                
            v = torch.stack([Normal(*q_C_T[task]).sample((ns_T,)).transpose(0, 1).reshape(z.size(0), ns_G*ns_T, -1)
                                 for task in q_C_T], 1)  
            
            r = self.encode_deterministic(X_C, Y_C, X_D)           
            p_Y = self.decode(X_D, v, r)

            return p_Y  


class STP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tasks = config.tasks
        
        # latent encoding path
        self.set_encoder = nn.ModuleList([SetEncoder(config.dim_x, config.dim_ys[task], config.dim_hidden,
                                                     config.module_sizes[0], config.module_sizes[1]) for task in self.tasks])
        self.task_encoder = nn.ModuleList([TaskEncoder(config.dim_hidden, hierarchical=False) for task in self.tasks])

        # deterministic encoding path
        self.conditional_set_encoder = nn.ModuleList([ConditionalSetEncoder(config.dim_x, config.dim_ys[task], config.dim_hidden,
                                                                            config.module_sizes[0], config.module_sizes[1]) for task in self.tasks])
        
        # decoding path
        self.decoder = nn.ModuleList([MTPDecoder(config.dim_x, config.dim_ys[task], config.dim_hidden, config.module_sizes[3])
                                      for task in self.tasks])
        
    def state_dict_(self):
        state_dict = {task: {} for task in self.tasks}
        for name, child in self.named_children():
            for t_idx, task in enumerate(self.tasks):
                state_dict[task][name] = child[t_idx].state_dict()
                
        return state_dict
    
    def state_dict_task(self, task):
        state_dict = {}
        for name, child in self.named_children():
            t_idx = self.tasks.index(task)
            state_dict[name] = child[t_idx].state_dict()
                
        return state_dict
    
    def load_state_dict_(self, state_dict):
        for name, child in self.named_children():
            for t_idx, task in enumerate(self.tasks):
                child[t_idx].load_state_dict(state_dict[task][name])
    
    def encode_task(self, X, Y):
        # task-specific latent in across-task inference of latent path
        q_T = {}
        for t_idx, task in enumerate(self.tasks):
            D_t = torch.cat((X[task], Y[task]), -1)
            s_t = self.set_encoder[t_idx](D_t)
            q_T[task] = self.task_encoder[t_idx](s_t)
            
        return q_T
    
    def encode_deterministic(self, X_C, Y_C, X_D):
        U_C = {}
        r = {}
        # cross-attention layers in across-task inference of deterministic path               
        for t_idx, task in enumerate(self.tasks):
            C_t = torch.cat((X_C[task], Y_C[task]), -1)
            U_C[task] = self.conditional_set_encoder[t_idx](C_t, X_C[task], X_D[task])
            # self-attention layers in across-task inference of deterministic path
            r[task] = U_C[task]
        return r
    
    def decode(self, X, v, r):
        if not self.training:
            X = {task: X[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks}  
            r = {task: r[task].unsqueeze(1).repeat(1, v.size(2), 1, 1) for task in self.tasks} 
        p_Y = {}
        
        for t_idx, task in enumerate(self.tasks):
            p_Y[task] = self.decoder[t_idx](X[task], v[:, t_idx], r[task])
            
        return p_Y
        
        
    def forward(self, X_C, Y_C, X_D, Y_D=None, ns_G = 1, ns_T=1): 
        if self.training:
            assert Y_D is not None
            q_C_T = self.encode_task(X_C, Y_C)
            q_D_T = self.encode_task(X_D, Y_D)
            v = torch.stack([Normal(*q_C_T[task]).rsample() for task in self.tasks], 1)

            r = self.encode_deterministic(X_C, Y_C, X_D)
            
            p_Y = self.decode(X_D, v, r)

            return p_Y, None, None, q_D_T, q_C_T
        else:                                                    #start to check this section during validation
            q_C_T = self.encode_task(X_C, Y_C)
            v = torch.stack([Normal(*q_C_T[task]).sample((ns_T,)).transpose(0, 1)
                                 for task in q_C_T], 1)          

            r = self.encode_deterministic(X_C, Y_C, X_D)

            p_Y = self.decode(X_D, v, r)

            return p_Y

        
        
        
        
        
        
        
        
        
        
        
        
        




        


