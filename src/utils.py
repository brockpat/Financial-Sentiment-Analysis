# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:51:12 2026

@author: patri
"""

# %% Libraries

import numpy as np
import pandas as pd


# %% Functions

def get_train_test_val_split(df, 
                             train_share = 0.7, val_share = 0.15, test_share = 0.15, 
                             seed = 42):
    
    # 1. Assert valid input
    assert (train_share + val_share + test_share) == 1.0, "Shares must sum to 1!"
    
    # 2. Set the seed for reproducibility
    np.random.seed(seed)
    
    # 3. Get all indices and shuffle them
    indices = df.index.tolist()
    np.random.shuffle(indices)
    
    # 4. Calculate split points
    n = len(indices)
    train_end = int(n * train_share)
    val_end = int(n * (train_share + val_share))
    
    # 5. Slice the shuffled indices into three lists
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    print(f"Train size: {len(train_idx)}")
    print(f"Val size:   {len(val_idx)}")
    print(f"Test size:  {len(test_idx)}")
    
    return train_idx, val_idx, test_idx