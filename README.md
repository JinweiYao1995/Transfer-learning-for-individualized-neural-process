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
| `config_mtp_test.yaml`       | Testing setting for MTNP                     |
| `config_single_iteration.yaml` | Training setting for ANP                  |
| `config_single_test.yaml`    | Testing setting for ANP                      |

---

## Folder Structure

### `dataset/`
- `init.py`: Initializes the data loader for creating batches of training and testing samples  
- `synthetic.py`: Preprocesses and reads data, and adjusts for data imbalance  
- `utils.py`: Defines the collator function to batch context and target sets and pass data into the CUDA environment

### `Experiments/`
- Stores trained models

### `signal_first/`
- Stores generated signals from `generate_data.py`

### `model/`
- `init.py`: Retrieves different models  
- `attention.py`: Creates the attention components  
- `mlp.py`: Creates multi-layer perceptron components  
- `Modules/`: Encoder and decoder components  
- `methods.py`: Constructs neural network structures for ANP, MTNP, proposed models, and proposed w/o source

### `Train/`
- `init.py`: Initializes the optimizer object  
- `loss.py`: Defines the loss objective function  
- `scheduler.py`: Custom scheduler for training Neural Processes  
- `utils.py`: Utility functions for storing training results

### Other Files
- `argument.py`: Sets up default arguments for constructing the experiments  
- `generate_data.py`: Constructs the signals in Setting I (Eq. 10 of the manuscript)  
- `Run_compare.R`: Implements MGP-based transfer learning in R  
- `TrainData.R`: Prepares the data in Signal Setting I in R

---

## License

This repository is intended for research and educational use. Please cite the original manuscript if you use this code.


