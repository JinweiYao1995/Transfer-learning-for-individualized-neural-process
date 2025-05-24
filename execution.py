# -*- coding: utf-8 -*-
"""
Created on Thu May 15 22:47:14 2025

@author: yaojinwei
"""


#Training for NP, MTNP, proposed and Proposed w/o sources
print("Running training...")
exec(open("main.py").read())
print("Training is completed successfully.")




#Training for MGP-based transfer learning 
import torch
import pandas as pd

X,Y,Xperm,Yperm,XD,YD = torch.load('signal_first/source1,source2,source3,target1_N100_n100.pth')

# Convert tensors to DataFrames and save to CSV
for key, tensor in Y.items():
    # Flatten the last dimension to make it compatible with CSV format
    df = pd.DataFrame(tensor.squeeze(-1).numpy())
    df.to_csv(f"{key}.csv", index=False)

print("Tensors saved as CSV files.")



import subprocess

print("Running MGP: Run_compare.R...")

# Run the R script and capture output
result = subprocess.run(["Rscript", "Run_compare.R"], capture_output=True, text=True)

print("MGP completed successfully...")

    
    

#Testing for all five methods

print("Running testing for all methods...")
exec(open("test.py").read())