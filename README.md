# Transfer Learning for Individualized Neural Processes

Simulation reproduction for the three-context RMSE boxplot of **Figure 5**
(Signal Setting I) of the manuscript. One command trains every method from
scratch and draws the figure:

```bash
python execution.py
```

---

## Methods compared

| Column in the figure | Model | Training |
|---|---|---|
| ANP | single-task Attentive Neural Process | two-stage baseline: one shared run on the pooled target data, then conditioned on each individual at test time |
| MTNP | Multi-Task Neural Process over sources + target | two-stage baseline (also serves as the KD teacher) |
| MTNP-KD | ANP-capacity student distilled from the frozen MTNP teacher (λ = 0.5, K = 20 moment-matched latent samples, forward KL) | one-stage: one model per test individual |
| Proposed | individualized transfer NP: sources + mean profile + the individual's context, fitted jointly | one-stage: one model per test individual |
| Proposed w/o source | ablation: mean profile only | one-stage: one model per test individual |
| Proposed w/o mean | ablation: raw target replications instead of the mean profile | one-stage: one model per test individual |
| MGP | MGP-based transfer learning (R implementation) | optional, see below |

Each individualized method is trained at context sizes 6 / 8 / 10 for all 80
test individuals; the boxplot groups the 80 RMSEs per context.

## Pipeline (`execution.py`)

The script runs six stages. **Every stage skips itself when its output already
exists**, so it is safe to interrupt and rerun; progress is printed to stdout.

1. **Data** — `generate_data.py` (seeded) draws the Setting-I signals
   (Eq. 10 of the manuscript) into `signal_first/`, including `raw.pth`
   (the w/o-mean variant, target replications tiled instead of averaged),
   and exports CSV copies for the R benchmark.
2. **Baseline training** — the two-stage baselines ANP
   (`config_single_iteration.yaml`) and MTNP (`config_mtp_RS.yaml`), each to
   `experiments/.../best_error.pth`.
3. **Individualized training** — the one-stage methods Proposed / w/o source /
   w/o mean / MTNP-KD at context 6, 8 and 10; checkpoints land in
   `experiments/<run>/<model>/checkpoint<i>/full_<cc><cc>.pth`.
4. **Baseline evaluation** — ANP and MTNP conditioned on the 80 test
   individuals at each context (cached in `experiments/baseline_rmse.pth`).
5. **MGP (optional)** — if `Rscript` is on the PATH, `MGP/Run_compare.R` fits
   the MGP transfer-learning benchmark on the exported CSVs and writes
   `MGP/down_original.Rdata`. Without R (or without `pyreadr` on the Python
   side) the figure is simply drawn without the MGP column.
6. **Figure** — `compare_3context.png`.

### Runtime

On a single modern GPU the two baseline trainings take on the order of an
hour; the individualized stage is the bulk of the cost (4 methods × 3 contexts
× 80 individuals × 200 steps) — expect **roughly a day** in total. The R stage
adds several more hours if enabled. For a quick end-to-end smoke test of the
plumbing:

```bash
MTNP_N_IND=2 MTNP_NSTEPS=5 python execution.py        # PowerShell: $env:MTNP_N_IND='2'; ...
```

Environment knobs: `MTNP_N_IND` (individuals, default 80), `MTNP_NSTEPS`
(shortens **every** training — smoke tests only), `MTNP_CONTEXTS`
(default `6,8,10`). After a smoke run, delete `experiments/` before the real
run so the shortened checkpoints are not mistaken for finished stages.

## Requirements

- Python ≥ 3.9 with `torch`, `pyyaml`, `easydict`, `matplotlib`, `pandas`,
  `tensorboard` (and `pyreadr` if you want the MGP column read back)
- CUDA is used when available; CPU works but is slow
- Optional, for the MGP benchmark: R with `Matrix`, `nloptr`, `minqa`,
  `optimx`, `rootSolve`, `nlme`, `mvtnorm`, `MASS`

## Repository layout

| Path | Contents |
|---|---|
| `execution.py` | the one-command pipeline described above |
| `generate_data.py` | Setting-I signal generation (sources, target, mean and raw variants) |
| `argument.py` | default experiment arguments |
| `configs/` | one YAML per method / evaluation setting (see below) |
| `dataset/` | data loading: preprocessing, imbalance handling, batch collator |
| `model/` | network components: attention, MLP, encoder/decoder modules |
| `model/methods.py` | the models — ANP (`STP`), MTNP (`MTP`), Proposed (`IMTP`), w/o source (`IMTPs`), MTNP-KD student (`MTNPKD`) |
| `train/` | losses, LR/beta schedulers, experiment utilities |
| `train/trainer.py` | `train_step`, `train_step_kd`, `moment_match_teacher`, evaluation |
| `MGP/` | the R implementation of the MGP benchmark: `Run_compare.R` (model fitting, reads the exported CSVs) and `TrainData.R` (R-side data preparation) |

### Configs

| File | Purpose |
|---|---|
| `config_single_iteration.yaml` | ANP training |
| `config_single_test.yaml` | ANP evaluation |
| `config_mtp_RS.yaml` | MTNP training (KD teacher) |
| `config_mtp_test.yaml` | MTNP evaluation / teacher-side contexts |
| `config_imtp.yaml` | Proposed |
| `config_imtp_single.yaml` | Proposed w/o source |
| `config_imtp_nosplit.yaml` | Proposed w/o mean (reads `signal_first/raw.pth`) |
| `config_kd.yaml` | MTNP-KD student (λ, warmup, K, teacher paths) |

## Reproducibility notes

- `generate_data.py` is seeded (`--seed 0`), and each training stage reseeds
  through `configure_experiment`, so reruns are deterministic up to GPU
  nondeterminism; the baseline (ANP / MTNP) evaluation samples one latent per
  episode, so its boxes vary slightly between machines.
- The exact RMSE values therefore differ marginally from the manuscript
  figure, but the method ordering and gaps reproduce.

## License

This repository is intended for research and educational use. Please cite the
original manuscript if you use this code.
