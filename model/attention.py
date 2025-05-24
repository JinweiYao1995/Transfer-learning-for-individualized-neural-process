# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:01:41 2024

@author: yaojinwei
"""

import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    '''
    Implementation of the Attention mechanism
    '''
    def __init__(self, dim, num_heads=4, ln = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_split = dim // num_heads #feature per heads
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.fc_o = nn.Linear(dim, dim, bias=False)
        
        self.activation =  nn.ReLU(inplace=True)
        self.attn_dropout = nn.Dropout(0.1)
        self.residual_dropout1 = nn.Dropout(0.1)
        self.residual_dropout2 = nn.Dropout(0.1)
        if ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

    def forward(self, Q, K, V=None, mask_Q=None, mask_K=None, get_attn=False):
        if V is None: V = K
        
        if mask_Q is not None:
            Q = Q.clone().masked_fill(mask_Q.unsqueeze(-1), 0)
        else:
            mask_Q = torch.zeros(*Q.size()[:2], device=Q.device)
        
        if mask_K is not None:
            K = K.clone().masked_fill(mask_K.unsqueeze(-1), 0)
            V = V.clone().masked_fill(mask_K.unsqueeze(-1), 0)
        else:
            mask_K = torch.zeros(*K.size()[:2], device=K.device)
        
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        mask = ~((1 - mask_Q.unsqueeze(-1).float()).bmm((1 - mask_K.unsqueeze(-1).float()).transpose(1, 2)).bool().repeat(self.num_heads, 1, 1))
        
        A = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim)  #here both Q and K should have batch size on the first. Q is m*dk, K is n*dk. 
        A = A.masked_fill(mask, -1e38)
        A = torch.softmax(A, 2)
        A = A.masked_fill(mask, 0)
            
        A = self.attn_dropout(A)
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)  #(batch_size, seq_length, num_heads * head_dim)
        
        O = Q + self.residual_dropout1(O)
        O = O if getattr(self, 'ln1', None) is None else masked_forward(self.ln1, O, mask_Q.bool(), self.dim) #return self.ln1 or None 
        O = O + self.residual_dropout2(self.activation(masked_forward(self.fc_o, O, mask_Q.bool(), self.dim)))
        O = O if getattr(self, 'ln2', None) is None else masked_forward(self.ln2, O, mask_Q.bool(), self.dim)
        
        if get_attn: #get_attn is asked here
            return O, A
        else:
            return O
        
class Mean(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=True)
    
        
class SelfAttention(nn.Module):
    def __init__(self, dim, n_layers, ln=False):
        super().__init__()        
        self.attentions = nn.ModuleList([Attention(dim, ln=ln) for _ in range(n_layers)])
        
    def forward(self, Q, mask=None, **kwargs):
        for attention in self.attentions:
            Q = attention(Q, Q, Q, mask_Q=mask, mask_K=mask, **kwargs)
        return Q
        
        
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, n_layers, ln=False):
        super().__init__()
        
        self.query_proj = nn.Linear(dim_q, dim_v)
        self.key_proj = nn.Linear(dim_k, dim_v)
        self.attentions = nn.ModuleList([Attention(dim_v, ln=ln) #dim_k = dim_v
                                         for _ in range(n_layers)])
        
    def forward(self, Q, K, V, **kwargs):
        Q = self.query_proj(Q)
        K = self.key_proj(K)
        for attention in self.attentions:
            Q = attention(Q, K, V, **kwargs)        #most likely it will not throw error.     
        return Q
    
def masked_forward(module, x, mask, out_dim, **kwargs):
    assert x.size()[:-1] == mask.size()             #matches the shape of the mask tensor
    out = torch.zeros(*mask.size(), out_dim).to(x.device) #Output Tensor Initialization:
    out[~mask] = module(x[~mask], **kwargs)    #Masked Forward Pass:
    
    return out
