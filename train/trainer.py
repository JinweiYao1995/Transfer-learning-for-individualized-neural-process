import torch
import torch.nn.functional as F
import warnings

from .loss import compute_elbo, compute_error, compute_normalized_nll
from .utils import broadcast_squeeze, broadcast_index , plot_curves #broadcast_mean
from dataset.utils import to_device
import copy



def train_step(model, optimizer, config, logger, *train_data):
    '''
    Perform a training step.
    '''
    # forward
    X_C, Y_C, X_D, Y_D, X_comp, Y_comp = train_data
    p_Y, q_D_G, q_C_G, q_D_T, q_C_T = model(X_C, Y_C, X_D, Y_D)
        
    loss = -compute_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T, config, logger)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # update global step
    logger.global_step += 1


@torch.no_grad()
def moment_match_teacher(teacher, X_C, Y_C, X_D, K):
    '''
    Frozen-teacher predictive distribution at the query inputs X_D, reduced to
    one Gaussian per point by Monte-Carlo moment matching over K latent
    samples. Returns (mu, var) for task 'target1' shaped (1, n, 1); the
    variance is floored at 1e-6.
    '''
    teacher.eval()
    p_Y = teacher(X_C, Y_C, X_D, ns_G=K, ns_T=1)
    mu_k, sd_k = p_Y['target1']                              # (1, K, n, 1)
    mu = mu_k.mean(dim=1)
    var = (sd_k.pow(2) + mu_k.pow(2)).mean(dim=1) - mu.pow(2)
    return mu, var.clamp_min(1e-6)


def train_step_kd(model, optimizer, config, logger, teacher_mu, teacher_var, *train_data):
    '''
    train_step + predictive-distribution distillation (MTNP-KD): the student
    fits its own episode (ELBO) and matches the teacher's per-point Gaussians
    at the episode's target inputs via forward KL(teacher || student), with a
    linear lambda warmup over the first kd_warmup_frac of training.
    '''
    X_C, Y_C, X_D, Y_D, X_comp, Y_comp = train_data
    p_Y, q_D_G, q_C_G, q_D_T, q_C_T = model(X_C, Y_C, X_D, Y_D)

    loss_np = -compute_elbo(Y_D, p_Y, q_D_G, q_C_G, q_D_T, q_C_T, config, logger)

    mu_s, sd_s = p_Y['target2']                              # (B, n, 1); teacher (1, n, 1) broadcasts
    var_s = sd_s.pow(2)
    kld = (torch.log(sd_s) - 0.5 * torch.log(teacher_var)
           + 0.5 * ((teacher_var + (teacher_mu - mu_s).pow(2)) / var_s - 1))
    loss_kd = kld.mean()

    warm = max(1, int(config.kd_warmup_frac * config.n_steps))
    lam = config.lambda_kd_max * min(1.0, logger.global_step / warm)
    loss = loss_np + lam * loss_kd

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.global_step += 1
    return loss_np.item(), loss_kd.item(), lam


@torch.no_grad()
def inference_map(model, *test_data):
    '''
    perform iterative prediction
    '''
    X_C, Y_C, X_D = test_data
    
    model.eval()
    p_Ys = model(X_C, Y_C, X_D) 
    model.train()
    
    return broadcast_squeeze(p_Ys, 1)



def evaluate(model, test_loader, device, config, logger=None,
             imputer=None, config_imputer=None, tag='valid'):
    '''
    Calculate error of model based on the posterior predictive mean.
    '''
    errors = {task: 0 for task in config.tasks}
    nlls = {task: 0 for task in config.tasks}    
    n_datasets = 0
    for b_idx, test_data in enumerate(test_loader):
        if config.data == 'synthetic':
            X_C, Y_C, X_D, Y_D, X_comp, Y_comp = to_device(test_data, device)
        else:
            raise NotImplementedError           
        # Prediction
        X_comp = {task: X_comp[task].expand(config.global_batch_size, 100, 1) for task in config.tasks}
        Y_D_pred_map = inference_map(model, X_C, Y_C, X_comp)    
            
       # compute errors
        nlls_ = compute_normalized_nll(Y_comp, Y_D_pred_map, config.task_types)  #two equation in loss, use MAP
        errors_ = compute_error(Y_comp, Y_D_pred_map, config.task_types)
        
        #batch denormalization
        for task in config.tasks:
            nlls[task] += (nlls_[task]*  config.global_batch_size)
            errors[task] += (errors_[task]*  config.global_batch_size)
        n_datasets +=  config.global_batch_size #24*24

    for task in config.tasks:
        nlls[task] /= n_datasets
        errors[task] /= n_datasets
        
    if logger is not None:
        for task in config.tasks:
            logger.writer.add_scalar(f'{tag}/nll_{task}', nlls[task].item(),
                                     global_step=logger.global_step)
            logger.writer.add_scalar(f'{tag}/error_{task}', errors[task].item(),
                                     global_step=logger.global_step)
        logger.writer.flush()
    
    return nlls, errors


def evaluate_test(model, device, config, individual = None, test_iterator = None):
    '''
    Evaluate both kinds of method: the two-stage baselines (ANP / MTNP,
    individual=None -- the shared model is conditioned on every test
    individual) and the one-stage individualized methods (individual=i).
    '''

    config = copy.deepcopy(config)
    config.global_batch_size = 1

    errors = {}
    nlls = {}
    #two-stage baselines: condition the shared model on each test individual
    if individual is None:
        X, Y, Xperm, Yperm, XD, YD = torch.load(config.data_path)
    
        for dataset in range(80):
            X_C = {}; Y_C = {}
            X_D = {}; Y_D = {}
            X_comp = {}; Y_comp = {}
            for task in config.tasks:
                cs = config.context_size[task]
                X_comp[task] = X[task].unsqueeze(0)
                Y_comp[task] = Y[task][dataset].unsqueeze(0)
                if task == 'target2':
                    ts = config.target_size[task]
                    X_C[task] = Xperm[task][dataset,0:cs].unsqueeze(0) #need to check how it respond whether it keeps the 1
                    Y_C[task] = Yperm[task][dataset,0:cs].unsqueeze(0)
                    X_D[task] = Xperm[task][dataset,0:ts].unsqueeze(0)
                    Y_D[task] = Yperm[task][dataset,0:ts].unsqueeze(0)
                else:
                    X_C[task] = Xperm[task][dataset,0:cs].unsqueeze(0) #need to check how it respond whether it keeps the 1
                    Y_C[task] = Yperm[task][dataset,0:cs].unsqueeze(0)
                #pass all variables into devices in CUDA        
                X_C = to_device(X_C,device); Y_C = to_device(Y_C,device)
                X_comp = to_device(X_comp,device); Y_comp = to_device(Y_comp,device)
            Y_D_pred_map = inference_map(model, X_C, Y_C, X_comp)    
            #if dataset in range(1):
            #    plot_curves(config.tasks, X_C, Y_C, X_comp, Y_comp, Y_D_pred_map, batch_size = 1)        
            #compute errors
            nlls[dataset] = compute_normalized_nll(Y_comp, Y_D_pred_map, config.task_types)  
            errors[dataset] = compute_error(Y_comp, Y_D_pred_map, config.task_types)           
        return nlls, errors
      
    else:
        #one-stage individualized methods: evaluate this individual's own model
        test_data = next(test_iterator)
        if config.data == 'synthetic':
              X_C, Y_C, X_D, Y_D, X_comp, Y_comp = to_device(test_data, device)
        else:
             raise NotImplementedError   
        for task in config.tasks:
                  X_C[task] = X_C[task][0].unsqueeze(0)
                  Y_C[task] = Y_C[task][0].unsqueeze(0)
                  X_D[task] = X_D[task][0].unsqueeze(0)
                  Y_D[task] = Y_D[task][0].unsqueeze(0)
                  X_comp[task] = X_comp[task].unsqueeze(0)
                  Y_comp[task] = Y_comp[task][0].unsqueeze(0)        
        Y_D_pred_map = inference_map(model, X_C, Y_C, X_comp)  #the model still only produce the last task   
        
        #if  individual in range(1):
        #    plot_curves(config.tasks, X_C, Y_C, X_comp, Y_comp, Y_D_pred_map, batch_size = 1)               
        
        nll = compute_normalized_nll(Y_comp, Y_D_pred_map, config.task_types)  
        error = compute_error(Y_comp, Y_D_pred_map, config.task_types)
        
        return nll, error

        
        
      
    
    
    
        
    
    

    
    

