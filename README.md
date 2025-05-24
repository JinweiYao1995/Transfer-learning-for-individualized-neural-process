# Transfer-learning-for-individualized-neural-process

Simulation Reproduction for Boxplot of Fig. 5 (Signal Setting I)

This repository provides code to reproduce the simulation results shown in Figure 5 of the manuscript under Signal Setting I. To run the simulation and generate the boxplot, execute:

python execution.py

------------------------------------------------------------------------

Implementation Overview

The execution.py script integrates:
- A training stage (via main.py)
- A testing stage (via test.py)

The training stage covers the following methods:
1. Attentive Neural Process (ANP)
2. Multi-Task Neural Process (MTNP)
3. Proposed method
4. Proposed method without source
5. MGP-based transfer learning

Note: Training takes approximately 15 hours for all methods combined. The testing stage executes almost instantly.

------------------------------------------------------------------------

Modify Simulation Settings

Configurations are located in the configs/ folder:

File                          | Purpose
------------------------------|-------------------------------------------------
config_imtp.yaml              | One-stage modeling for Proposed
config_imtp_single.yaml      | One-stage modeling for Proposed w/o source
config_mtp_RS.yaml           | Training setting for MTNP
config_mtp_test.yaml         | Testing setting for MTNP
config_single_iteration.yaml | Training setting for ANP
config_single_test.yaml      | Testing setting for ANP

------------------------------------------------------------------------

Project Structure

dataset/
  - __init__.py: Initializes data loaders
  - synthetic.py: Reads and balances data
  - utils.py: Collates context and target sets, transfers to CUDA

Experiments/
  - Stores trained models

signal_first/
  - Stores signals generated from generate_data.py

model/
  - __init__.py: Loads model components
  - attention.py: Defines attention modules
  - mlp.py: Multi-layer perceptron components
  - Modules/: Encoder and decoder components
  - methods.py: Neural network structures for ANP, MTNP, proposed models

Train/
  - __init__.py: Initializes optimizers
  - loss.py: Defines loss functions
  - scheduler.py: Custom learning rate scheduler
  - utils.py: Utility functions for training logs and outputs

------------------------------------------------------------------------

Additional Scripts

- argument.py: Sets up default arg
