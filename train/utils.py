import os
import sys
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .schedulers import LRScheduler, HPScheduler

def configure_experiment(config, args):
    # update config with arguments
    config.model = args.model
    config.seed = args.seed
    config.name_postfix = args.name_postfix
    #config.pma = args.pma
    
    # parse arguments
    if args.n_steps > 0: config.n_steps = args.n_steps
    if args.lr > 0: config.lr = args.lr
    if args.global_batch_size > 0: config.global_batch_size = args.global_batch_size
    if args.dim_hidden > 0: config.dim_hidden = args.dim_hidden
        
    if args.lr_schedule != '': config.lr_schedule = args.lr_schedule
    if args.beta_T_schedule != '': config.beta_T_schedule = args.beta_T_schedule
    if args.beta_G_schedule != '': config.beta_G_schedule = args.beta_G_schedule

    
    # set seeds
    torch.backends.cudnn.deterministic = True  #ensure the same output 
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # for debugging
    if args.debug_mode:
        config.n_steps = 3
        config.log_iter = 1
        config.val_iter = 1
        config.save_iter = 1
        config.imputer_path = config.imputer_path.replace(config.log_dir, config.log_dir + '_debugging')
        config.log_dir += '_debugging'

    # set directories
    if args.log_root != '':
        config.log_root = args.log_root
    if args.name != '':
        exp_name = args.name
    else:
        exp_name = config.model + config.name_postfix
    #if args.imputer_path != '':
    #    config.imputer_path = args.imputer_path
        
    os.makedirs(config.log_root, exist_ok=True)
    os.makedirs(os.path.join(config.log_root, config.log_dir), exist_ok=True)
    os.makedirs(os.path.join(config.log_root, config.log_dir, exp_name), exist_ok=True)
    log_dir = os.path.join(config.log_root, config.log_dir, exp_name, 'logs')
    save_dir = os.path.join(config.log_root, config.log_dir, exp_name, 'checkpoints')
    #if os.path.exists(save_dir):
    #    shutil.rmtree(save_dir)
    #os.makedirs(save_dir)

    # tensorboard logger
    logger = Logger(log_dir, config.tasks)
    log_keys = ['nll_normalized'] + [f'nll_{task}' for task in config.tasks]
    if config.model in ['mtp']:
        log_keys.append('kld_G')
    if config.model in ['imtp','imtps','mtp','stp']:
        log_keys += ['kld_T_normalized'] + [f'kld_{task}' for task in config.tasks]
    for log_key in log_keys:
        logger.register_key(log_key)
    
    return logger, save_dir, log_keys


def get_schedulers(optimizer, config):
    lr_scheduler = LRScheduler(optimizer, config.lr_schedule, config.lr, config.n_steps, config.lr_warmup)
    beta_G_scheduler = beta_T_scheduler = None
    if config.model in ['mtp']:    
        beta_G_scheduler = HPScheduler(config, 'beta_G', config.beta_G_schedule, config.beta_G, config.n_steps, config.beta_G_warmup)
    if config.model in ['mtp','stp', 'imtps','imtp']:    
        beta_T_scheduler = HPScheduler(config, 'beta_T', config.beta_T_schedule, config.beta_T, config.n_steps, config.beta_T_warmup)
    
    return lr_scheduler, beta_G_scheduler, beta_T_scheduler


# Define the plot_curves function with various parameters
def plot_curves(tasks, X_C, Y_C, X_comp, Y_comp, Y_D_pred, size=3, markersize=5, batch_size = 1, n_row=None):
    plt.rc('xtick', labelsize=3*size)  # Set the size of x-axis tick labels
    plt.rc('ytick', labelsize=3*size)  # Set the size of y-axis tick labels
    
        
    n_row = len(tasks)  # Ensure number of rows does not exceed the number of subplots
    n_line = (len(tasks) // n_row) * n_row  # Adjust number of subplots to be a multiple of rows
    
    for idx_sub in range(batch_size):
        plt.figure(figsize=(size * n_row * 4 / 3, size * (n_line // n_row)))  # Create a figure for the task
        for idx_task, task in enumerate(tasks): 
            plt.subplot(n_line//n_row, n_row, idx_task + 1)  # Create a subplot
            x_d = X_comp[task][idx_sub,].cpu().squeeze(-1)  # Get corresponding X_D values and move to CPU
            y_d = Y_comp[task][idx_sub,].cpu().squeeze(-1)  # Get corresponding Y_D values and move to CPU
            p_D = torch.argsort(x_d)  # Sort x_d indices

            line1, = plt.plot(x_d[p_D], y_d[p_D], color= 'k', alpha = 0.5)  # Plot the target values

            x_c = X_C[task][idx_sub, ].cpu().squeeze(-1)
            y_c = Y_C[task][idx_sub, ].cpu().squeeze(-1)
            p_C = torch.argsort(x_c)  # Sort x_c indices
            plt.scatter(x_c[p_C], y_c[p_C], color= 'b', alpha = 1, s=markersize * 6)  #plot the context values
           
            
            mu = Y_D_pred[task][0][idx_sub,].cpu().squeeze(-1)
            sigma = Y_D_pred[task][1][idx_sub,].cpu().squeeze(-1)
            line2, = plt.plot(x_d[p_D], mu[p_D], color= 'r') #label=f'{nll:.3f}')
            plt.fill_between(x_d[p_D], mu[p_D] - sigma[p_D], mu[p_D] + sigma[p_D], color='r', alpha=0.2)
            
            
            plt.legend([line1, line2], ['Underlying Signal with noised coefficients', 'Prediction'], fontsize='small')
            
            #plt.ylim(-1.5, 2)  # Adjust these limits as needed
            plt.title(f'Task {task}')
        plt.tight_layout()
        plt.show()
            

    

class Logger():
    def __init__(self, log_dir, tasks, reset=True):
        if os.path.exists(log_dir) and reset:
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        
        self.logs = {}
        self.logs_saved = {}
        self.iters = {}
        
        self.key = 0
        self.value = 0
        
    def register_key(self, key):
        self.logs[key] = 0
        self.logs_saved[key] = 0
        self.iters[key] = 0
        
    def add_value(self, key, value):
        
        self.key = key
        self.value = value
        
        self.logs[key] += value
        self.iters[key] += 1
        
    def get_value(self, key):
        if self.iters[key] == 0:
            return self.logs_saved[key]
        else:
            return self.logs[key] / self.iters[key]
        
    def reset(self, keys):
        for key in keys:
            self.logs_saved[key] = self.get_value(key)
            self.logs[key] = 0
            self.iters[key] = 0
            
    def log_values(self, keys, pbar=None, tag='train', global_step=0):
        if pbar is not None:
            desc = 'step {:05d}'.format(global_step)
            
            if 'nll_normalized' in keys:
                desc += ', {}: {:.3f}'.format('nll_norm', self.get_value('nll_normalized'))
            if 'kld_T_normalized' in keys:
                desc += ', {}: {:.3f}'.format('kld_T_norm', self.get_value('kld_T_normalized'))
            if 'kld_G' in keys:
                desc += ', {}: {:.3f}'.format('kld_G', self.get_value('kld_G'))
            pbar.set_description(desc)

        for key in filter(lambda x: x not in ['nll_normalized', 'kld_T_normalized'], keys):
            self.writer.add_scalar('{}/{}'.format(tag, key), self.get_value(key), global_step=global_step)
            
        for key in filter(lambda x: x in ['nll_normalized', 'kld_T_normalized', 'kld_G'], keys):
            self.writer.add_scalar('{}_summary/{}'.format(tag, key), self.get_value(key), global_step=global_step)
            

class Saver:
    def __init__(self, model, save_dir, config):
        self.save_dir = save_dir
        self.config = config
        self.tasks = config.tasks
        self.model_type = config.model
        if self.model_type == 'stp':
            self.best_nll_state_dict = model.state_dict_()
            self.best_error_state_dict = model.state_dict_()        
        self.best_nll = float('inf')
        self.best_nlls = {task: float('inf') for task in config.tasks}
        self.best_error = float('inf')
        self.best_errors = {task: float('inf') for task in config.tasks}
        
    def save(self, model, valid_nlls, valid_errors, global_step, save_name):
        torch.save({'model': model.state_dict_(), 'config': self.config,
                    'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                    os.path.join(self.save_dir, save_name))
        
                
    def save_best(self, model, valid_nlls, valid_errors, global_step):
        valid_nll = sum([valid_nlls[task] for task in self.tasks])
        valid_error = sum([valid_errors[task] for task in self.tasks])
        
        # save best model
        if self.model_type == 'stp':
            update_nll = False
            update_error = False
            for task in self.best_nlls:
                if valid_nlls[task] < self.best_nlls[task]:
                    self.best_nlls[task] = valid_nlls[task]
                    self.best_nll_state_dict[task] = model.state_dict_task(task)
                    update_nll = True
                    
                if valid_errors[task] < self.best_errors[task]:
                    self.best_errors[task] = valid_errors[task]
                    self.best_error_state_dict[task] = model.state_dict_task(task)
                    update_error = True
                    
            if update_nll:
                torch.save({'model': self.best_nll_state_dict, 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_nll.pth'))
            if update_error:
                torch.save({'model': self.best_error_state_dict, 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_error.pth'))
        else:
            if valid_nll < self.best_nll:
                self.best_nll = valid_nll
                torch.save({'model': model.state_dict_(), 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_nll.pth'))
            if valid_error < self.best_error:
                self.best_error = valid_error
                torch.save({'model': model.state_dict_(), 'config': self.config,
                            'nlls': valid_nlls, 'errors': valid_errors, 'global_step': global_step},
                            os.path.join(self.save_dir, 'best_error.pth'))
    



def broadcast_squeeze(data, dim):
    def squeeze_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.squeeze(dim)
        elif isinstance(data, tuple):
            return tuple(map(squeeze_wrapper, data))
        elif isinstance(data, list):
            return list(map(squeeze_wrapper, data))
        elif isinstance(data, dict):
            return {key: squeeze_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
    
    return squeeze_wrapper(data)


def broadcast_index(data, idx):
    def index_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, tuple):
            return data[idx]
        elif isinstance(data, list):
            return data[idx]
        elif isinstance(data, dict):
            return {key: index_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError
    
    return index_wrapper(data)
