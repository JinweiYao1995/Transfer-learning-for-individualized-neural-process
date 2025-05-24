import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch

from dataset import load_data
from model import get_model
from train import train_step, evaluate, configure_experiment, get_schedulers, Saver




#Train of single-task ANP model 


# ENVIRONMENTAL SETTINGSs
torch.set_num_threads(1)

# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_single_iteration.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'stp'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# load train and valid data
train_loader, train_iterator, train_entire = load_data(config, device)

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)


#set saving file directory
checkpoint_num = 1
next_checkpoint_dir = f"checkpoint{checkpoint_num}"
save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# checkpoint saver
saver = Saver(model, save_dir, config_copy) 
    
# MAIN LOOP
pbar = tqdm.tqdm(total=config.n_steps, initial=0,
                 bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")

# Control train step
while logger.global_step < config.n_steps:
    
    #Pass into the next iteration (context set and target set)
    train_data = next(train_iterator)
    train_step(model, optimizer, config, logger, *train_data)
    
    # schedulers step
    lr_scheduler.step()
    if config.model in ['mtp']:
        beta_G_scheduler.step()
    if config.model in ['imtp','imtps','stp','mtp']:
        beta_T_scheduler.step()
    
    # logging that record the convergence progress
    if logger.global_step % config.log_iter == 0: 
        logger.log_values(log_keys, pbar, 'train', logger.global_step)
        logger.reset(log_keys)
        logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
        if config.model in ['mtp']:
            logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
        if config.model in ['imtp','imtps','stp','mtp']:
            logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

    # evaluate and visualize
    if logger.global_step % config.val_iter == 0:
        valid_nlls, valid_errors = evaluate(model, train_loader, device, config, logger, tag='valid')
        saver.save_best(model, valid_nlls, valid_errors, logger.global_step)

    
    # save model
    if logger.global_step % config.save_iter == 0:
        # save current model
        saver.save(model, valid_nlls, valid_errors, logger.global_step, f'step_{logger.global_step:06d}.pth')
                    
    pbar.update(1)
    
#Save Model and Terminate.
saver.save(model, valid_nlls, valid_errors, logger.global_step, 'last.pth')
    
pbar.close()

####################################################################################


#Train of Multi-task NP model 

import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch

from dataset import load_data
from model import get_model
from train import train_step, evaluate, configure_experiment, get_schedulers, Saver


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)


# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_mtp_RS.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'mtp'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# load train and valid data
train_loader, train_iterator, train_entire = load_data(config, device)

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)


#set saving file directory
checkpoint_num = 1
next_checkpoint_dir = f"checkpoint{checkpoint_num}"
save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# checkpoint saver
saver = Saver(model, save_dir, config_copy) 
    
# MAIN LOOP
pbar = tqdm.tqdm(total=config.n_steps, initial=0,
                 bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")


while logger.global_step < config.n_steps:
    #Pass into the next iteration (context set and target set)
    train_data = next(train_iterator)
    train_step(model, optimizer, config, logger, *train_data)
    
    # schedulers step
    lr_scheduler.step()
    if config.model in ['mtp']:
        beta_G_scheduler.step()
    if config.model in ['imtp','imtps','stp','mtp']:
        beta_T_scheduler.step()
    
    # logging
    if logger.global_step % config.log_iter == 0: #log_iter is 100
        logger.log_values(log_keys, pbar, 'train', logger.global_step)
        logger.reset(log_keys)
        logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
        if config.model in ['mtp']:
            logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
        if config.model in ['imtp','imtps','stp','mtp']:
            logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

    # evaluate and visualize
    if logger.global_step % config.val_iter == 0:
        valid_nlls, valid_errors = evaluate(model, train_loader, device, config, logger, tag='valid')  #inputer will be None
        saver.save_best(model, valid_nlls, valid_errors, logger.global_step)

    
    # save model
    if logger.global_step % config.save_iter == 0:
        # save current model
        saver.save(model, valid_nlls, valid_errors, logger.global_step, f'step_{logger.global_step:06d}.pth')
                    
    pbar.update(1)
    
# Save Model and Terminate.
saver.save(model, valid_nlls, valid_errors, logger.global_step, 'last.pth')
    
pbar.close()








##################Training for the Proposed with context = 4 in the individual function of interest 
import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch
import time

from dataset import load_data
from model import get_model
from train import train_step, configure_experiment, get_schedulers, Saver
from train.trainer import evaluate_test


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)


# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_imtp.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'imtp'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)

config.context_size['target2'] = 6
config.target_size['target2'] = 6


#iterate over different individual function of interest 
for individual_id in range(80):
    print(f"individual_id {individual_id}")
    checkpoint_num = individual_id+1
    next_checkpoint_dir = f"checkpoint{checkpoint_num}"
    save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader, train_iterator, train_entire = load_data(config, device, individual = individual_id)
    
    # MAIN LOOP
    pbar = tqdm.tqdm(total=config.n_steps, initial=0,
         bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    while logger.global_step < config.n_steps: #total 30000 ste
    # train step
        train_data = next(train_iterator)
        train_step(model, optimizer, config, logger, *train_data)
    
        # schedulers step
        lr_scheduler.step()
     
        # logging
        if logger.global_step % config.log_iter == 0: #log_iter is 100
            logger.log_values(log_keys, pbar, 'train', logger.global_step)
            logger.reset(log_keys)
            logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
            if config.model in ['mtp']:
                logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
                if config.model in ['imtp','imtps','stp','mtp']:
                    logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

        # evaluate and visualize
        if logger.global_step % config.val_iter == 0:  
            valid_loader, valid_iterator, valid_entire = load_data(config, device, split ='ind_valid', individual = individual_id)            
            valid_nlls, valid_errors = evaluate_test(model, device, config, individual_id, valid_iterator)  #inputer will be None

        # save model
        if logger.global_step % config.save_iter == 0:
            training_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler,
                'valid_nlls': valid_nlls,
                'valid_errors': valid_errors,
                'global_step': logger.global_step}
            checkfile = os.path.join(save_dir, 'full_0606.pth')
            torch.save(training_state, checkfile)
        pbar.update(1)   
    pbar.close()
    logger.global_step = 0






##################Training for the Proposed with context = 8 in the individual function of interest 

import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch
import time

from dataset import load_data
from model import get_model
from train import train_step, configure_experiment, get_schedulers, Saver
from train.trainer import evaluate_test


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)


# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_imtp.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'imtp'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)

config.context_size['target2'] = 8
config.target_size['target2'] = 8


#iterate over different individual function of interest 
for individual_id in range(80):
    print(f"individual_id {individual_id}")
    checkpoint_num = individual_id+1
    next_checkpoint_dir = f"checkpoint{checkpoint_num}"
    save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    train_loader, train_iterator, train_entire = load_data(config, device, individual = individual_id)
    
    # MAIN LOOP
    pbar = tqdm.tqdm(total=config.n_steps, initial=0,
         bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    while logger.global_step < config.n_steps: #total 30000 ste
    # train step
        train_data = next(train_iterator)
        train_step(model, optimizer, config, logger, *train_data)
    
        # schedulers step
        lr_scheduler.step()
     
        # logging
        if logger.global_step % config.log_iter == 0: #log_iter is 100
            logger.log_values(log_keys, pbar, 'train', logger.global_step)
            logger.reset(log_keys)
            logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
            if config.model in ['mtp']:
                logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
                if config.model in ['imtp','imtps','stp','mtp']:
                    logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

        # evaluate and visualize
        if logger.global_step % config.val_iter == 0:  
            valid_loader, valid_iterator, valid_entire = load_data(config, device, split ='ind_valid', individual = individual_id)            
            valid_nlls, valid_errors = evaluate_test(model, device, config, individual_id, valid_iterator)  #inputer will be None
            
        # save model
        if logger.global_step % config.save_iter == 0:
            training_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler,
                'valid_nlls': valid_nlls,
                'valid_errors': valid_errors,
                'global_step': logger.global_step}
            checkfile = os.path.join(save_dir, 'full_0808.pth')
            torch.save(training_state, checkfile)
          
        pbar.update(1)   
    pbar.close()
    logger.global_step = 0


##################Training for the Proposed with context = 10 in the individual function of interest 

import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch
import time

from dataset import load_data
from model import get_model
from train import train_step, configure_experiment, get_schedulers, Saver
from train.trainer import evaluate_test


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)


# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_imtp.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'imtp'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)

config.context_size['target2'] = 10
config.target_size['target2'] = 10


#iterate over different individual function of interest 
for individual_id in range(80):
    print(f"individual_id {individual_id}")
    checkpoint_num = individual_id+1
    next_checkpoint_dir = f"checkpoint{checkpoint_num}"
    save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    train_loader, train_iterator, train_entire = load_data(config, device, individual = individual_id)
    
    # MAIN LOOP
    pbar = tqdm.tqdm(total=config.n_steps, initial=0,
         bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    while logger.global_step < config.n_steps: #total 30000 ste
    # train step
        train_data = next(train_iterator)
        train_step(model, optimizer, config, logger, *train_data)
    
        # schedulers step
        lr_scheduler.step()
     
        # logging
        if logger.global_step % config.log_iter == 0: #log_iter is 100
            logger.log_values(log_keys, pbar, 'train', logger.global_step)
            logger.reset(log_keys)
            logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
            if config.model in ['mtp']:
                logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
                if config.model in ['imtp','imtps','stp','mtp']:
                    logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

        # evaluate and visualize
        if logger.global_step % config.val_iter == 0:  
            valid_loader, valid_iterator, valid_entire = load_data(config, device, split ='ind_valid', individual = individual_id)            
            valid_nlls, valid_errors = evaluate_test(model, device, config, individual_id, valid_iterator)  #inputer will be None

        # save model
        if logger.global_step % config.save_iter == 0:
            training_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler,
                'valid_nlls': valid_nlls,
                'valid_errors': valid_errors,
                'global_step': logger.global_step}
            checkfile = os.path.join(save_dir, 'full_1010.pth')
            torch.save(training_state, checkfile)
          
        pbar.update(1)   
    pbar.close()
    logger.global_step = 0









##################Training for the Proposed w/o sources with context = 4 in the individual function of interest 


import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch
import time

from dataset import load_data
from model import get_model
from train import train_step, configure_experiment, get_schedulers, Saver
from train.trainer import evaluate_test


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)


# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_imtp_single.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'imtps'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)

config.context_size['target2'] = 6
config.target_size['target2'] = 6



#iterate over differerent individual function of interest 
for individual_id in range(80):
    print(f"individual_id {individual_id}")
    checkpoint_num = individual_id+1
    next_checkpoint_dir = f"checkpoint{checkpoint_num}"
    save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    train_loader, train_iterator, train_entire = load_data(config, device, individual = individual_id)
    
    # MAIN LOOP
    pbar = tqdm.tqdm(total=config.n_steps, initial=0,
         bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    while logger.global_step < config.n_steps: #total 30000 ste
    # train step
        train_data = next(train_iterator)
        train_step(model, optimizer, config, logger, *train_data)
    
        # schedulers step
        lr_scheduler.step()
     
        # logging
        if logger.global_step % config.log_iter == 0: #log_iter is 100
            logger.log_values(log_keys, pbar, 'train', logger.global_step)
            logger.reset(log_keys)
            logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
            if config.model in ['mtp']:
                logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
                if config.model in ['imtp','imtps','stp','mtp']:
                    logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

        # evaluate and visualize
        if logger.global_step % config.val_iter == 0:  
            valid_loader, valid_iterator, valid_entire = load_data(config, device, split ='ind_valid', individual = individual_id)            
            valid_nlls, valid_errors = evaluate_test(model, device, config, individual_id, valid_iterator)  #inputer will be None
        # save model
        if logger.global_step % config.save_iter == 0:
            training_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler,
                'valid_nlls': valid_nlls,
                'valid_errors': valid_errors,
                'global_step': logger.global_step}
            checkfile = os.path.join(save_dir, 'full_0606.pth')
            torch.save(training_state, checkfile)
            
        pbar.update(1)  
    pbar.close()
    logger.global_step = 0
            


##################Training for the Proposed w/o sources with context = 8 in the individual function of interest 

import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch
import time

from dataset import load_data
from model import get_model
from train import train_step, configure_experiment, get_schedulers, Saver
from train.trainer import evaluate_test


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)


# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_imtp_single.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'imtps'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)

config.context_size['target2'] = 8
config.target_size['target2'] = 8



#iterate over differerent individual function of interest 
for individual_id in range(80):
    print(f"individual_id {individual_id}")
    checkpoint_num = individual_id+1
    next_checkpoint_dir = f"checkpoint{checkpoint_num}"
    save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    train_loader, train_iterator, train_entire = load_data(config, device, individual = individual_id)
    
    # MAIN LOOP
    pbar = tqdm.tqdm(total=config.n_steps, initial=0,
         bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    while logger.global_step < config.n_steps: #total 30000 ste
    # train step
        train_data = next(train_iterator)
        train_step(model, optimizer, config, logger, *train_data)
    
        # schedulers step
        lr_scheduler.step()
     
        # logging
        if logger.global_step % config.log_iter == 0: #log_iter is 100
            logger.log_values(log_keys, pbar, 'train', logger.global_step)
            logger.reset(log_keys)
            logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
            if config.model in ['mtp']:
                logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
                if config.model in ['imtp','imtps','stp','mtp']:
                    logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

        # evaluate and visualize
        if logger.global_step % config.val_iter == 0:  
            valid_loader, valid_iterator, valid_entire = load_data(config, device, split ='ind_valid', individual = individual_id)            
            valid_nlls, valid_errors = evaluate_test(model, device, config, individual_id, valid_iterator)  #inputer will be None
        # save model
        if logger.global_step % config.save_iter == 0:
            training_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler,
                'valid_nlls': valid_nlls,
                'valid_errors': valid_errors,
                'global_step': logger.global_step}
            checkfile = os.path.join(save_dir, 'full_0808.pth')
            torch.save(training_state, checkfile)
            
        pbar.update(1)  
    pbar.close()
    logger.global_step = 0
            



##################Training for the Proposed w/o sources with context = 10 in the individual function of interest 


import os
import tqdm
import copy
import yaml
from easydict import EasyDict

import torch
import time

from dataset import load_data
from model import get_model
from train import train_step, configure_experiment, get_schedulers, Saver
from train.trainer import evaluate_test


# ENVIRONMENTAL SETTINGSs
# to prevent over-threading
torch.set_num_threads(1)


# parse arguments
from argument import args


# load config
with open(os.path.join('configs', args.data, 'config_imtp_single.yaml'), 'r') as f:
    config = EasyDict(yaml.safe_load(f))
    
args.model = 'imtps'

# configure settings, logging and checkpointing paths
logger, save_dir, log_keys = configure_experiment(config, args)
config_copy = copy.deepcopy(config)
 
# set device
device = torch.device('cuda')

# model, optimizer, and schedulers
model = get_model(config, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, config)

config.context_size['target2'] = 10
config.target_size['target2'] = 10



#iterate over differerent individual function of interest 
for individual_id in range(80):
    print(f"individual_id {individual_id}")
    checkpoint_num = individual_id+1
    next_checkpoint_dir = f"checkpoint{checkpoint_num}"
    save_dir = os.path.join('experiments',config.log_dir, args.model, next_checkpoint_dir)
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    train_loader, train_iterator, train_entire = load_data(config, device, individual = individual_id)
    
    # MAIN LOOP
    pbar = tqdm.tqdm(total=config.n_steps, initial=0,
         bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    while logger.global_step < config.n_steps: #total 30000 ste
    # train step
        train_data = next(train_iterator)
        train_step(model, optimizer, config, logger, *train_data)
    
        # schedulers step
        lr_scheduler.step()
     
        # logging
        if logger.global_step % config.log_iter == 0: #log_iter is 100
            logger.log_values(log_keys, pbar, 'train', logger.global_step)
            logger.reset(log_keys)
            logger.writer.add_scalar('train/lr', lr_scheduler.lr, logger.global_step)
            if config.model in ['mtp']:
                logger.writer.add_scalar('train/beta_G', config.beta_G, logger.global_step)
                if config.model in ['imtp','imtps','stp','mtp']:
                    logger.writer.add_scalar('train/beta_T', config.beta_T, logger.global_step)

        # evaluate and visualize
        if logger.global_step % config.val_iter == 0:  
            valid_loader, valid_iterator, valid_entire = load_data(config, device, split ='ind_valid', individual = individual_id)            
            valid_nlls, valid_errors = evaluate_test(model, device, config, individual_id, valid_iterator)
        # save model
        if logger.global_step % config.save_iter == 0:
            training_state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler,
                'valid_nlls': valid_nlls,
                'valid_errors': valid_errors,
                'global_step': logger.global_step}
            checkfile = os.path.join(save_dir, 'full_1010.pth')
            torch.save(training_state, checkfile)
            
        pbar.update(1)  
    pbar.close()
    logger.global_step = 0
            








