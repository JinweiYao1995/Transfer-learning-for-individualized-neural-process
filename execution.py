# -*- coding: utf-8 -*-
"""
One-command reproduction of the three-context RMSE boxplot (Signal Setting I).

    python execution.py

The pipeline trains everything from scratch in six stages; every stage skips
itself when its output already exists on disk, so the script is safe to
interrupt and rerun:

  1. generate the synthetic signals               -> signal_first/*.pth (+ *.csv for R)
  2. train the two-stage baselines ANP and MTNP   -> experiments/runs_single_iteration,
     (stage one: fit the shared model on the         experiments/runs_mtp_RS
      pooled data; stage two happens at test
      time, conditioning on each individual)
  3. train the one-stage individualized methods at context sizes 6 / 8 / 10
     (each individual's model is fitted in a single stage):
       Proposed              (imtp,  mean profile + sources)
       Proposed w/o source   (imtps, mean profile only)
       Proposed w/o mean     (mtp on raw.pth: raw target curves, no mean profile)
       MTNP-KD               (ANP-capacity student distilled from the frozen
                              MTNP teacher, lambda = 0.5)
  4. evaluate ANP / MTNP on the 80 test individuals at context 6 / 8 / 10
  5. MGP-based transfer learning in R (optional -- needs Rscript on PATH; the
     figure is drawn without the MGP column when R or pyreadr is unavailable)
  6. draw compare_3context.png

Environment knobs (all optional):
  MTNP_N_IND    number of test individuals for the individualized methods (default 80)
  MTNP_NSTEPS   override n_steps of EVERY training (smoke tests only)
  MTNP_CONTEXTS comma list of context sizes (default '6,8,10')
Smoke test of the full plumbing:  MTNP_N_IND=2  MTNP_NSTEPS=5  python execution.py
"""
import matplotlib
matplotlib.use('Agg')   # headless: evaluate_test calls plot_curves(); an interactive
                        # backend would block on plt.show()
import os
import sys
import copy
import time
import shutil
import subprocess

import yaml
import torch
from easydict import EasyDict

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # all paths are repo-relative
sys.path.insert(0, os.getcwd())

from dataset import load_data
from dataset.utils import to_device
from model import get_model
from train import train_step, evaluate, configure_experiment, get_schedulers, Saver
from train.trainer import evaluate_test, train_step_kd, moment_match_teacher
from argument import args

torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_IND = int(os.environ.get('MTNP_N_IND', '80'))
NSTEPS_OVERRIDE = os.environ.get('MTNP_NSTEPS', '')
CONTEXTS = [int(c) for c in os.environ.get('MTNP_CONTEXTS', '6,8,10').split(',') if c.strip()]

DATA_NAME = 'source1,source2,source3,target1_N100_n100'
KD_LAMBDA = 0.5          # distillation weight used for the reported MTNP-KD results
BASELINE_RMSE = os.path.join('experiments', 'baseline_rmse.pth')
MGP_RDATA = 'down_original.Rdata'
FIGURE = 'compare_3context.png'

METHODS = ['ANP', 'MTNP', 'MTNP-KD', 'Proposed', 'Proposed w/o source',
           'Proposed w/o mean', 'MGP']
COLORS = {'ANP': 'lightblue', 'MTNP': 'lightgreen', 'MTNP-KD': 'lightgray',
          'Proposed': 'lightcoral', 'Proposed w/o source': 'lightsalmon',
          'Proposed w/o mean': 'khaki', 'MGP': 'plum'}
# training config and checkpoint layout of each one-stage individualized method
ONE_STAGE = {
    'Proposed':            ('config_imtp.yaml',         'imtp',   'runs_imtp'),
    'Proposed w/o source': ('config_imtp_single.yaml',  'imtps',  'runs_imtp_single'),
    'Proposed w/o mean':   ('config_imtp_nosplit.yaml', 'mtp',    'runs_mtp_individual'),
}


def log(msg):
    print(msg, flush=True)


def load_cfg(name):
    with open(os.path.join('configs', name)) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.num_workers = 0     # the per-individual loop re-creates DataLoaders;
                            # worker processes deadlock on Windows
    if NSTEPS_OVERRIDE:     # smoke-test knob: shortens EVERY training
        cfg.n_steps = int(NSTEPS_OVERRIDE)
    return cfg


# ---------------------------------------------------------------------------
# stage 1: data
# ---------------------------------------------------------------------------
def ensure_data():
    if not os.path.exists(os.path.join('signal_first', 'raw.pth')):
        log('[data] generating signals (generate_data.py, seed 0)')
        subprocess.run([sys.executable, 'generate_data.py'], check=True)
    else:
        log('[data] signal_first/ already populated -> skip')
    # CSV export of the same signals for the R (MGP) benchmark
    if not os.path.exists('target2.csv'):
        import pandas as pd
        _, Y, _, _, _, _ = torch.load(os.path.join('signal_first', f'{DATA_NAME}.pth'))
        for key, tensor in Y.items():
            pd.DataFrame(tensor.squeeze(-1).numpy()).to_csv(f'{key}.csv', index=False)
        log('[data] CSVs for the R benchmark written')


# ---------------------------------------------------------------------------
# stage 2: two-stage baselines (ANP = stp, MTNP = mtp) -- their first stage
# trains the shared model once; the second stage is test-time conditioning
# ---------------------------------------------------------------------------
def train_baseline(config_name, model_name):
    cfg = load_cfg(config_name)
    args.model = model_name
    done = os.path.join('experiments', cfg.log_dir, model_name, 'checkpoint1', 'best_error.pth')
    if os.path.exists(done):
        log(f'[baseline] {model_name} already trained ({done}) -> skip')
        return
    logger, _, _ = configure_experiment(cfg, args)
    model = get_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler, beta_G_scheduler, beta_T_scheduler = get_schedulers(optimizer, cfg)
    train_loader, train_iterator, _ = load_data(cfg, device)

    save_dir = os.path.join('experiments', cfg.log_dir, model_name, 'checkpoint1')
    os.makedirs(save_dir, exist_ok=True)
    saver = Saver(model, save_dir, copy.deepcopy(cfg))

    log(f'[baseline] {model_name} start (n_steps={cfg.n_steps})')
    t0 = time.time()
    every = max(1, cfg.n_steps // 10)
    while logger.global_step < cfg.n_steps:
        train_step(model, optimizer, cfg, logger, *next(train_iterator))
        lr_scheduler.step()
        if beta_G_scheduler is not None and cfg.model == 'mtp':
            beta_G_scheduler.step()
        if beta_T_scheduler is not None:
            beta_T_scheduler.step()
        s = logger.global_step
        if s % every == 0:
            r = s / max(1e-9, time.time() - t0)
            eta = (cfg.n_steps - s) / r if r > 0 else 0
            log(f'[baseline] {model_name} {s}/{cfg.n_steps}  {r:.2f} it/s  ETA {int(eta // 60)}m{int(eta % 60):02d}s')
    valid_nlls, valid_errors = evaluate(model, train_loader, device, cfg, logger, tag='valid')
    saver.save_best(model, valid_nlls, valid_errors, logger.global_step)
    saver.save(model, valid_nlls, valid_errors, logger.global_step, 'last.pth')
    log(f'[baseline] {model_name} done -> {save_dir}')


# ---------------------------------------------------------------------------
# stage 3: one-stage individualized methods, trained per test individual
# ---------------------------------------------------------------------------
def train_individuals(config_name, model_name, run_dir, context):
    cfg = load_cfg(config_name)
    args.model = model_name
    fname = f'full_{context:02d}{context:02d}.pth'
    last = os.path.join('experiments', run_dir, model_name, f'checkpoint{N_IND}', fname)
    if os.path.exists(last):
        log(f'[individual] {model_name} c{context} already has {N_IND} checkpoints -> skip')
        return
    logger, _, _ = configure_experiment(cfg, args)
    model = get_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler, _, _ = get_schedulers(optimizer, cfg)

    cfg.context_size['target2'] = context   # trained directly at this context
    cfg.target_size['target2'] = context

    log(f'[individual] {model_name} c{context} start: {N_IND} individuals, n_steps={cfg.n_steps}')
    for individual_id in range(N_IND):
        save_dir = os.path.join('experiments', run_dir, model_name, f'checkpoint{individual_id+1}')
        os.makedirs(save_dir, exist_ok=True)
        _, train_iterator, _ = load_data(cfg, device, individual=individual_id)
        logger.global_step = 0
        while logger.global_step < cfg.n_steps:
            train_step(model, optimizer, cfg, logger, *next(train_iterator))
            lr_scheduler.step()
        _, valid_iterator, _ = load_data(cfg, device, split='ind_valid', individual=individual_id)
        valid_nlls, valid_errors = evaluate_test(model, device, cfg, individual_id, valid_iterator)
        torch.save({'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'lr_scheduler_state': lr_scheduler,
                    'valid_nlls': valid_nlls,
                    'valid_errors': valid_errors,
                    'global_step': logger.global_step},
                   os.path.join(save_dir, fname))
        log(f'[individual] {model_name} c{context} individual {individual_id+1}/{N_IND}  '
            f'RMSE={valid_errors["target2"].item():.4f}')
    log(f'[individual] {model_name} c{context} done')


def train_individuals_kd(context):
    '''
    MTNP-KD: mirrors train_individuals, plus a per-individual setup step that
    caches the frozen teacher's soft targets (fixed for a given individual and
    context because the episode is deterministic).
    '''
    cfg = load_cfg('config_kd.yaml')
    cfg.lambda_kd_max = KD_LAMBDA
    args.model = 'mtnpkd'
    fname = f'full_{context:02d}{context:02d}.pth'
    last = os.path.join('experiments', cfg.log_dir, 'mtnpkd', f'checkpoint{N_IND}', fname)
    if os.path.exists(last):
        log(f'[individual] mtnpkd c{context} already has {N_IND} checkpoints -> skip')
        return
    logger, _, _ = configure_experiment(cfg, args)
    model = get_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler, _, beta_T_scheduler = get_schedulers(optimizer, cfg)

    cfg.context_size['target2'] = context
    cfg.target_size['target2'] = context

    # frozen teacher; its source contexts follow the MTNP evaluation convention
    # (test file, per-individual row)
    tcfg = EasyDict(yaml.safe_load(open(cfg.teacher_config)))
    tck = torch.load(cfg.teacher_ckpt, map_location=device)
    teacher = get_model(tck['config'], device)
    teacher.load_state_dict_(tck['model'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    _, _, tXperm, tYperm, _, _ = torch.load(tcfg.data_path)

    log(f'[individual] mtnpkd c{context} start: {N_IND} individuals, n_steps={cfg.n_steps}, lambda={KD_LAMBDA}')
    for individual_id in range(N_IND):
        save_dir = os.path.join('experiments', cfg.log_dir, 'mtnpkd', f'checkpoint{individual_id+1}')
        os.makedirs(save_dir, exist_ok=True)
        _, train_iterator, train_entire = load_data(cfg, device, individual=individual_id)

        # teacher soft targets, computed ONCE per individual: the target1 slot
        # gets the student's OWN context so C^T' matches point-for-point;
        # queries = the episode's target inputs X_D
        c = cfg.context_size['target2']
        sX_C = train_entire.Xperm['target2'][0][:c].unsqueeze(0)
        sY_C = train_entire.Yperm['target2'][0][:c].unsqueeze(0)
        sX_D = train_entire.XD['target2'].unsqueeze(0)
        X_Ct, Y_Ct, X_Dt = {}, {}, {}
        for task in teacher.tasks:
            if task == 'target1':
                X_Ct[task], Y_Ct[task], X_Dt[task] = sX_C, sY_C, sX_D
            else:
                cs = tcfg.context_size[task]
                X_Ct[task] = tXperm[task][individual_id, :cs].unsqueeze(0)
                Y_Ct[task] = tYperm[task][individual_id, :cs].unsqueeze(0)
                X_Dt[task] = X_Ct[task]          # dummy queries; outputs unused
        X_Ct = to_device(X_Ct, device); Y_Ct = to_device(Y_Ct, device); X_Dt = to_device(X_Dt, device)
        tea_mu, tea_var = moment_match_teacher(teacher, X_Ct, Y_Ct, X_Dt, cfg.kd_K)

        logger.global_step = 0
        while logger.global_step < cfg.n_steps:
            train_step_kd(model, optimizer, cfg, logger, tea_mu, tea_var, *next(train_iterator))
            lr_scheduler.step()
            beta_T_scheduler.step()
        _, valid_iterator, _ = load_data(cfg, device, split='ind_valid', individual=individual_id)
        valid_nlls, valid_errors = evaluate_test(model, device, cfg, individual_id, valid_iterator)
        torch.save({'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'lr_scheduler_state': lr_scheduler,
                    'valid_nlls': valid_nlls,
                    'valid_errors': valid_errors,
                    'global_step': logger.global_step},
                   os.path.join(save_dir, fname))
        log(f'[individual] mtnpkd c{context} individual {individual_id+1}/{N_IND}  '
            f'RMSE={valid_errors["target2"].item():.4f}')
    log(f'[individual] mtnpkd c{context} done')


# ---------------------------------------------------------------------------
# stage 4: baseline (ANP / MTNP) evaluation at every context
# ---------------------------------------------------------------------------
def eval_baselines():
    if os.path.exists(BASELINE_RMSE):
        log(f'[eval] {BASELINE_RMSE} exists -> skip')
        return torch.load(BASELINE_RMSE)
    out = {}
    for method, cfg_name, exp in (('ANP', 'config_single_test.yaml', 'stp'),
                                  ('MTNP', 'config_mtp_test.yaml', 'mtp')):
        with open(os.path.join('configs', cfg_name)) as f:
            tcfg = EasyDict(yaml.safe_load(f))
        ck = torch.load(os.path.join('experiments', tcfg.eval_dir, exp, 'checkpoint1', 'best_error.pth'),
                        map_location=device)
        model = get_model(ck['config'], device)
        model.load_state_dict_(ck['model'])
        out[method] = {}
        for c in CONTEXTS:
            tcfg.context_size['target1'] = c
            _, errors = evaluate_test(model, device, tcfg)
            out[method][c] = [v['target1'].item() for _, v in errors.items()]
            log(f'[eval] {method} c{c}: n={len(out[method][c])}')
    torch.save(out, BASELINE_RMSE)
    return out


# ---------------------------------------------------------------------------
# stage 5: MGP benchmark in R (optional)
# ---------------------------------------------------------------------------
def run_mgp():
    if os.path.exists(MGP_RDATA):
        log(f'[MGP] {MGP_RDATA} exists -> skip')
        return True
    if shutil.which('Rscript') is None:
        log('[MGP] Rscript not found on PATH -> the figure is drawn WITHOUT the MGP column')
        return False
    log('[MGP] running Run_compare.R (this is the slowest R stage; expect hours)')
    result = subprocess.run(['Rscript', 'Run_compare.R'], capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(MGP_RDATA):
        log('[MGP] Run_compare.R failed -> the figure is drawn WITHOUT the MGP column')
        log(result.stderr[-2000:])
        return False
    return True


# ---------------------------------------------------------------------------
# stage 6: the figure
# ---------------------------------------------------------------------------
def ckpt_rmse(run_dir, sub, context):
    vals = []
    for n in range(1, N_IND + 1):
        p = os.path.join('experiments', run_dir, sub, f'checkpoint{n}', f'full_{context:02d}{context:02d}.pth')
        if os.path.exists(p):
            vals.append(torch.load(p, map_location='cpu')['valid_errors']['target2'].item())
    return vals


def build_figure(baselines, mgp_ok):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    raw = {c: {} for c in CONTEXTS}
    for c in CONTEXTS:
        raw[c]['ANP'] = baselines['ANP'][c]
        raw[c]['MTNP'] = baselines['MTNP'][c]
        raw[c]['MTNP-KD'] = ckpt_rmse('runs_kd', 'mtnpkd', c)
        for m, (_, sub, run_dir) in ONE_STAGE.items():
            raw[c][m] = ckpt_rmse(run_dir, sub, c)
    methods = list(METHODS)
    if mgp_ok:
        try:
            import pyreadr
            r = pyreadr.read_r(MGP_RDATA)
            for c in CONTEXTS:
                raw[c]['MGP'] = r[f'RMSE{c}'].squeeze().tolist()
        except ImportError:
            log('[figure] pyreadr not installed -> MGP column dropped')
            methods.remove('MGP')
    else:
        methods.remove('MGP')

    data, positions, box_colors, group_centers = [], [], [], []
    pos, gap_in, gap_between = 1.0, 0.6, 1.7
    for c in CONTEXTS:
        start = pos
        for m in methods:
            data.append(raw[c][m]); positions.append(pos); box_colors.append(COLORS[m])
            pos += gap_in
        group_centers.append((start + pos - gap_in) / 2)
        pos += gap_between

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    LW = 0.9
    box = plt.boxplot(data, positions=positions, patch_artist=True,
                      boxprops=dict(linewidth=LW), whiskerprops=dict(linewidth=LW),
                      capprops=dict(linewidth=LW), flierprops=dict(markeredgewidth=LW))
    for patch, c_ in zip(box['boxes'], box_colors):
        patch.set_facecolor(c_)
    for med in box['medians']:
        med.set(color='black', linewidth=LW * 1.6)

    plt.ylabel('RMSE')
    plt.xticks(group_centers, [f'context = {c}' for c in CONTEXTS])
    plt.tick_params(axis='x', which='both', bottom=False)
    plt.xlim(min(positions) - 0.8, max(positions) + 0.8)
    plt.legend(handles=[mpatches.Patch(color=COLORS[m], label=m) for m in methods],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=15)
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=200, bbox_inches='tight')
    log(f'[figure] saved {FIGURE}')

    import statistics as st
    for c in CONTEXTS:
        log(f'[figure] context {c} means: ' +
            '  '.join(f'{m}={st.mean(raw[c][m]):.3f}' for m in methods if raw[c][m]))


if __name__ == '__main__':
    log(f'device: {device}   individuals: {N_IND}   contexts: {CONTEXTS}')
    ensure_data()

    train_baseline('config_single_iteration.yaml', 'stp')   # ANP
    train_baseline('config_mtp_RS.yaml', 'mtp')             # MTNP (also the KD teacher)

    for c in CONTEXTS:
        for m, (cfg_name, sub, run_dir) in ONE_STAGE.items():
            train_individuals(cfg_name, sub, run_dir, context=c)
        train_individuals_kd(context=c)

    baselines = eval_baselines()
    mgp_ok = run_mgp()
    build_figure(baselines, mgp_ok)
    log('all done.')
