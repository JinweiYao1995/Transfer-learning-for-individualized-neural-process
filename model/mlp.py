# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class FFB(nn.Module):
    def __init__(self, dim_in, dim_out, ln): #ln is layerNorm   
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.LayerNorm(dim_out) if ln else nn.Identity(),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.layers(x)

    
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers, ln=False):
        super().__init__()
        assert n_layers >= 1
        
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        layers = []
        for l_idx in range(n_layers):
            di = dim_in if l_idx == 0 else dim_hidden
            do = dim_out if l_idx == n_layers - 1 else dim_hidden
            layers.append(FFB(di, do, ln))
            
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class LatentMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, n_layers=2, ln = False):
        super().__init__()
        
        assert n_layers >= 1
        if n_layers >= 2:
            self.mlp = MLP(dim_in, dim_hidden, dim_hidden, n_layers-1, ln)
        else:
            self.mlp = None
        self.hidden_to_mu = nn.Linear(dim_hidden, dim_out)
        self.hidden_to_log_sigma = nn.Linear(dim_hidden, dim_out)
        
    def forward(self, x):
        hidden = self.mlp(x) if self.mlp is not None else x        
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_log_sigma(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        return mu, sigma