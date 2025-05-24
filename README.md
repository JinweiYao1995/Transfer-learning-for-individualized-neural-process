# Transfer-learning-for-individualized-neural-process

Simulation Reproduction for Boxplot of Fig. 5 (Signal Setting I)

This repository provides code to reproduce the simulation results shown in Figure 5 of the manuscript under Signal Setting I. To run the simulation and generate the boxplot, execute:

**python execution.py**

------------------------------------------------------------------------

Implementation Overview

The execution.py script integrates:
- A training stage (via **main.py**)
- A testing stage (via **test.py**)

The training stage covers the following methods:
1. Attentive Neural Process (ANP)
2. Multi-Task Neural Process (MTNP)
3. Proposed method
4. Proposed method without source
5. MGP-based transfer learning

Note: Training takes approximately **16 hours** for all methods combined (80 RMSE). The testing stage executes almost instantly.

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

Except for the mentioned files, the demo folder contains the following additional files:

1. Folder **dataset**:
   1.1 init.py: Initializes the data loader for creating batches of training and testing samples
   1.2 synthetic.py: Preprocesses and reads data, and adjusts for data imbalance
   1.3 utils.py: Defines the collator function to batch context and target sets and pass data into the CUDA environment

2. Folder **Experiments**: Path reserved for storing trained models

3. Folder **signal_first**: Path reserved for storing generated signals from 'generate_data.py'

4. Folder **model**:
   4.1 init.py: Retrieves different models
   4.2 attention.py: Creates the attention components
   4.3 mlp.py: Creates multi-layer perceptron components
   4.4 Modules: Creates encoder and decoder components
   4.5 methods.py: Constructs neural network structures for ANP, MTNP, the proposed model, and the proposed model without sources

5. Folder **Train**:
   5.1 init.py: Initializes the optimizer object
   5.2 loss.py: Defines the loss objective function
   5.3 scheduler.py: Custom scheduler for training Neural Processes
   5.4 utils.py: Utility functions for storing training results

6. argument.py: Sets up default arguments for constructing the experiments

7. generate_data.py: Constructs the signals in Setting I (Eq. 10 of the manuscript)

8. Run_compare.R: Implements MGP-based transfer learning in R

9. TrainData.R: Prepares the data in Signal Setting I in R
