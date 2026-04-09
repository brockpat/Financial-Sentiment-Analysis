# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:41:57 2026

@author: patri
"""

# %% Libraries

import pandas as pd 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import src.utils as utils

# %% Create the Model

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        
        # Initialise Model Weights
        self.linear = nn.Linear(input_dim, 1)
    
    # Forward pass
    def forward(self, x):
        z = self.linear(x)
        return torch.sigmoid(z)
    
def train_logistic_model(df, epochs=200):
    print("\n--- Starting Logistic Regression Training ---")
    train_idx, val_idx, test_idx = utils.get_train_test_val_split(df)

    # Prepare Tensors
    X_train = torch.tensor(np.stack(df.loc[train_idx, 'embedding'].values), dtype=torch.float32)
    y_train = torch.tensor(df.loc[train_idx, 'sentiment'].values, dtype=torch.float32).unsqueeze(1)

    X_val = torch.tensor(np.stack(df.loc[val_idx, 'embedding'].values), dtype=torch.float32)
    y_val = torch.tensor(df.loc[val_idx, 'sentiment'].values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

    # Input Dimension & Model
    input_dim = X_train.shape[1] 
    model = LogisticRegressionModel(input_dim)
    
    # Define Loss and Optimiser
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Initial Accuracy Check (Before Training)
    model.eval() # Set to evaluation mode
    with torch.no_grad():
        # We use the validation set to see how random weights perform
        initial_val_outputs = model(X_val)
        initial_preds = (initial_val_outputs > 0.5).float()
        initial_acc = (initial_preds == y_val).float().mean()
        
    print(f"--- Pre-training Baseline ---")
    print(f"Initial Val Acc: {initial_acc:.4f}")
    print(f"-----------------------------\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            # Convert probabilities to binary predictions
            preds = (val_outputs > 0.5).float()
            acc = (preds == y_val).float().mean()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")

    return model 