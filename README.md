# Transfer-learning-for-individualized-neural-process

## Simulation Reproduction for Boxplot of Fig. 5 (Signal Setting I)

This repository provides code to reproduce the simulation results shown in **Figure 5** of the manuscript under **Signal Setting I**. To run the simulation and generate the boxplot, execute:

```bash
python execution.py
```

---

## Implementation Overview

The `execution.py` script integrates:
- A training stage (via **main.py**)
- A testing stage (via **test.py**)

The training stage covers the following methods:
1. Attentive Neural Process (ANP)
2. Multi-Task Neural Process (MTNP)
3. Proposed method
4. Proposed method without source
5. MGP-based transfer learning

> **Note**: Training takes approximately **16 hours** for all methods combined (80 RMSE). The testing stage executes almost instantly.

---

## Modify Simulation Settings

Configurations are located in the `configs/` folder:

| File                          | Purpose                                      |
|------------------------------|----------------------------------------------|
| `config_imtp.yaml`           | One-stage modeling for Proposed              |
| `config_imtp_single.yaml`    | One-stage modeling for Proposed w/o source   |
| `config_mtp_RS.yaml`         | Training setting for MTNP                    |
|

