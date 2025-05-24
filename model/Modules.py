# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:05:35 2024

@author: yaojinwei
"""

import torch
import torch.nn as nn

from model.mlp import MLP, LatentMLP
from model.attention import SelfAttention, CrossAttention, masked_forward  


class SetEncoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, mlp_layers, attn_layers):  
        super().__init__()
        self.dim_hidden = dim_hidden
        
        self.mlp = MLP(dim_x + dim_y, dim_hidden, dim_hidden, mlp_layers)      
        self.task_embedding = nn.Parameter(torch.randn(dim_hidden), requires_grad=True) 

        self.attention = SelfAttention(dim_hidden, attn_layers, ln= False)       
       
    def forward(self, C):                                     
        # nan mask
        mask = C[..., -1].isnan()   #creates a mask based on the last dimension
        
        # project (x, y) to s
        s = masked_forward(self.mlp, C, mask, self.dim_hidden) # (B, n, h)
        
        # add task embedding e^t
        s = s + self.task_embedding.unsqueeze(0).unsqueeze(1)
        
        # intra-task attention
        s = self.attention(s, mask=mask) # (B, n, h)
            
        # intra-task aggregation
        #s = self.pool(s).squeeze(1) # (B, h)
        s = s.mean(dim = 1) # (B, h)
        
        return s
    
    
class GlobalEncoder(nn.Module):
    def __init__(self, dim_hidden, attn_layers):
        super().__init__()
        self.attention = SelfAttention(dim_hidden, attn_layers,ln= False)
        self.global_amortizer = LatentMLP(dim_hidden, dim_hidden, dim_hidden, 2, ln = False)

    def forward(self, s):
        # inter-task attention
        s = self.attention(s) # (B, T, h)
        
        # inter-task aggregation
        s = s.mean(dim = 1) # (B, h)
        
        self.sg = s 
        
        # global latent distribution
        q_G = self.global_amortizer(s)
        
        return q_G
    
    
class TaskEncoder(nn.Module):
    def __init__(self, dim_hidden, hierarchical=True):
        super().__init__()
        self.hierarchical = hierarchical
        self.task_amortizer = LatentMLP(dim_hidden*(1 + int(hierarchical)), dim_hidden, dim_hidden, 2, ln = False)
        
    def forward(self, s, z=None):
        # hierarchical conditioning
        if self.hierarchical:
            assert z is not None
            s = torch.cat((s, z), -1)
            
        # task latent distribution
        q_T = self.task_amortizer(s)   
        return q_T    

    
    
class ConditionalSetEncoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, mlp_layers, attn_layers):
        super().__init__()
        self.dim_hidden = dim_hidden
        
        self.mlp = MLP(dim_x + dim_y, dim_hidden, dim_hidden, mlp_layers, ln= False)
        self.task_embedding = nn.Parameter(torch.randn(dim_hidden), requires_grad=True)
        self.attention = CrossAttention(dim_x, dim_x, dim_hidden, attn_layers, ln = False)
        
    def forward(self, C, X_C, X_D):
        # nan mask
        mask = C[..., -1].isnan()
        
        # project (x, y) to s
        d = masked_forward(self.mlp, C, mask, self.dim_hidden) # (B, n, h)
        
        # add task embedding e^t
        d = d + self.task_embedding.unsqueeze(0).unsqueeze(1)
        
        # intra-task attention
        u = self.attention(X_D, X_C, d, mask_K=mask)  #target inputs, context_inputs, and local d
        
        return u
    
class MultiTaskAttention(nn.Module):
    def __init__(self, dim_hidden, att_layers,  ln=False):
        super().__init__()   
        self.attention = SelfAttention(dim_hidden, att_layers, ln = ln)
        
    def forward(self, Q):
        bs, nb, ts, _ = Q.size()  
        Q_ = Q.transpose(1, 2).reshape(bs*ts, nb, -1)
        Q_ = self.attention(Q_) 
        Q = Q_.reshape(bs, ts, *Q_.size()[1:]).transpose(1, 2) 
        return Q


class MultiTaskPooling(nn.Module):
    def __init__(self, dim_hidden, att_layers,  ln=False):
        super().__init__()   
        self.attention = SelfAttention(dim_hidden, att_layers, ln = ln)
        
    def forward(self, Q):
        bs, nb, ts, _ = Q.size()  
        Q_ = Q.transpose(1, 2).reshape(bs*ts, nb, -1)
        Q_ = self.attention(Q_) 
        Q = Q_.reshape(bs, ts, *Q_.size()[1:]).transpose(1, 2)  
        Q = Q.mean(dim = 1)
        return Q
    

class MTPDecoder(nn.Module):
    def __init__(self, dim_x, dim_y, dim_hidden, n_layers):
        super().__init__()
        self.input_projection = nn.Linear(dim_x, dim_hidden)
        self.task_embedding = nn.Parameter(torch.randn(dim_hidden), requires_grad=True)
        self.output_amortizer = LatentMLP(dim_hidden*3, dim_y, dim_hidden, n_layers, ln = False) 
        
    def forward(self, X, v, r):
        # project x to w
        w = self.input_projection(X) 
        
        if self.training:
            # add task embedding e^t
            w = w + self.task_embedding.unsqueeze(0).unsqueeze(1)
        else:
            # add task embedding e^t
            w = w + self.task_embedding.unsqueeze(0).unsqueeze(1).unsqueeze(2)    
            
        # concat w, v, r
        v = v.unsqueeze(-2).repeat(*([1]*(len(w.size())-2)), w.size(-2), 1)
        
        decoder_input = torch.cat((w, v, r), -1)
        
        # output distribution
        p_Y = self.output_amortizer(decoder_input)
        
        return p_Y  #which has mu and sigma 
    
    
    
    
    



