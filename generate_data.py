# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 04:01:18 2024

@author: yaojinwei
"""
import torch
import os


#Create the signal shows in Fig.4 and Eq.10 of signal setup 1. 

tasks = ['source1', 'source2', 'source3', 'target1']
activations = {
    'source1': lambda x: torch.sin(x),          #0
    'source2': lambda x: torch.tanh(2*x) ,         #torch.tanh(x)  #0
    'source3': lambda x: torch.sigmoid(x),     #torch.sigmoid(x)   #0.2
    'target1': lambda x: torch.exp(-x.pow(2))     #torch.exp(-x.pow(2))  
}
colors = {
    'source1': 'r',
    'source2': 'g',
    'source3': 'b', 
    'target1': 'c',
}

target_size = {'source1': 40, 'source2': 40 , 'source3': 40, 'target1': 10} #if different then, reset when create test data

def generate_data(n_functions, n_points, independent=True):
    Y = {task: [] for task in tasks}
    Xperm = {task:[] for task in tasks}
    Yperm = {task:[] for task in tasks}
    pp_target = {}               
    
    x = torch.linspace(-5, 5, n_points).view(n_points,1) # -5 to +5
    X = {task: x for task in tasks}
        
    if not independent:
        print("we don't consider dependent signals")
            
    #preset a collection of target points and then change the order
    for task in tasks:
        pp_target[task] = torch.linspace(0, n_points-1, steps= target_size[task]).round().int()
    
                
    for dataset in range(n_functions):
        for task in tasks: 
            a_ = torch.exp(torch.rand(1, 1) - 0.5) 
            w_ = torch.exp(torch.rand(1, 1) - 0.5)
            b_ = 2*torch.rand(1, 1) - 1  
            c_ = 2*torch.rand(1, 1) - 1 
            y = a_ * activations[task](w_ * x + b_) + c_             
            Y[task].append(y)
            #arrange permutation of context points in Xperm and Yperm
            ppi = torch.randperm(len(pp_target[task]))
            pp = pp_target[task][ppi]
            Xperm[task].append(x[pp])
            Yperm[task].append(y[pp])
    Xperm = {task: torch.stack(Xperm[task]) for task in tasks}
    Yperm = {task: torch.stack(Yperm[task]) for task in tasks}
    Y = {task: torch.stack(Y[task]) for task in tasks}        
    XD = {task: X[task][pp_target[task]] for task in tasks}
    YD = {task: Y[task][:,pp_target[task]] for task in tasks}
    
    return X, Y, Xperm, Yperm, XD, YD
    #returns the entire underlying truth (X,Y), permutated (XD,YD) and the origianl (XD,YD) 
    


if __name__ == '__main__':
    import argparse

    #create the training data set
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_functions', type=int, default=100)
    parser.add_argument('--n_points', type=int, default=100)
    parser.add_argument('--independent', '-ind', default=True, action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)   # reproducible signals

    #create first sample dataset
    X, Y, Xperm, Yperm, XD, YD = generate_data(args.n_functions, args.n_points, args.independent)
    
    #create the test dataset, where the function is 40
    n_functions = 80
    
    X1,Y1,Xperm1,Yperm1, XD1, YD1 = generate_data(n_functions, args.n_points, args.independent)
    
    #Include the individual function for training
    X['target2'] = X1['target1'] 
    Y['target2'] = Y1['target1'] 
    Xperm['target2'] = Xperm1['target1'] 
    Yperm['target2'] = Yperm1['target1'] 
    XD['target2'] = XD1['target1'] 
    YD['target2'] = YD1['target1'] 
    
 
    data_dir = 'signal_first'
    name = ','.join(tasks) + f'_N{args.n_functions}_n{args.n_points}'
    os.makedirs(data_dir, exist_ok=True)
    
    torch.save((X1, Y1, Xperm1, Yperm1, XD1, YD1), os.path.join(data_dir, f'test{name}.pth'))
    #The training data now has 4+1 = 5 tasks.
    
    torch.save((X, Y, Xperm, Yperm, XD, YD), os.path.join(data_dir, f'{name}.pth')) 
    #The testing data still has 4 tasks 
 
    #Create the sample means for target and context points. 
    n_functions = 100  
    sample_counts = [100,100,100,5]
    pp_target = {}
    Yperm_mean = {task:[] for task in tasks}
    Xperm_mean = {task:[] for task in tasks}
    XDmean = {}; YDmean = {}
    Y_comp_m = {task:[] for task in tasks}
    
    for idx_task, task in enumerate(tasks):   #so total available only has 4 tasks  
        pp_target[task] = torch.linspace(0, args.n_points-1, steps= target_size[task]).round().int()
        #window = sample_counts[idx_task]-1
        YDmean[task] = torch.mean(Y[task][0:sample_counts[idx_task],pp_target[task]], dim = 0)   
        Y_comp_m[task] = torch.mean(Y[task][0:sample_counts[idx_task]], dim = 0).expand(n_functions,-1,-1)  
        XDmean[task] = X[task][pp_target[task]]
    #Permutate the target points
    for  task in tasks:
        for dataset in range(n_functions):
            ppi = torch.randperm(len(pp_target[task]))
            Yperm_mean[task].append(YDmean[task][ppi])
            Xperm_mean[task].append(XDmean[task][ppi])
    
    Xperm_mean = {task: torch.stack(Xperm_mean[task]) for task in tasks} 
    Yperm_mean = {task: torch.stack(Yperm_mean[task]) for task in tasks}        
    XDmean = {task:X[task][pp_target[task]] for task in tasks}
    YDmean = {task:Y_comp_m[task][:,pp_target[task]] for task in tasks}
    
    
    #inherit the  indivdiuals to the training data 
    Xperm_mean['target2'] = Xperm['target2'] 
    Yperm_mean['target2'] = Yperm['target2'] 
    Y_comp_m['target2'] = Y['target2'] 
    XDmean['target2'] = XD['target2']
    YDmean['target2'] = YD['target2']
    
    
    torch.save((X, Y_comp_m, Xperm_mean, Yperm_mean, XDmean, YDmean), os.path.join(data_dir, f'average{name}.pth'))

    #X, Y are the original data, with the replication as the original data
    #Xperm and Yperm are randomly orderred target for generating XC and YC, with the replication as the original data
    #XD and YD are unorderred target points, with the replication as the original data


    # raw<name>.pth: the "Proposed w/o mean" data. Same tasks as the average
    # file, but target1 keeps the sample_counts[-1]=5 RAW curves the mean was
    # computed from, tiled deterministically (row i <- curve i % 5) up to the
    # sources' 100 rows, instead of being replaced by their mean. target2 (the
    # 80 test individuals) is carried over unchanged.
    N_RAW = sample_counts[-1]
    tile = torch.arange(args.n_functions) % N_RAW
    Y_raw = {t: Y[t].clone() for t in Y}
    Xperm_raw = {t: Xperm[t].clone() for t in Xperm}
    Yperm_raw = {t: Yperm[t].clone() for t in Yperm}
    YD_raw = {t: YD[t].clone() for t in YD}
    Y_raw['target1'] = Y['target1'][tile]
    Xperm_raw['target1'] = Xperm['target1'][tile]
    Yperm_raw['target1'] = Yperm['target1'][tile]
    YD_raw['target1'] = YD['target1'][tile]
    torch.save((X, Y_raw, Xperm_raw, Yperm_raw, XD, YD_raw), os.path.join(data_dir, 'raw.pth'))

    print('saved', data_dir, ':', f'{name}.pth,', f'test{name}.pth,', f'average{name}.pth,', 'raw.pth')
    

    


            