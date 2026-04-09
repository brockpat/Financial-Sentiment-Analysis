# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:54:13 2026

@author: patri
"""

# %% Libraries
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import torch
import pandas as pd
import numpy as np 

from dotenv import load_dotenv
from openai import OpenAI

# Import project modules
from src.pipeline import prepare_data
from src.model_logistic import train_logistic_model
from src.test_data import prepare_handcrafted_test_data

# %% Main

def main():
    load_dotenv()
    data_path = BASE_DIR / "data" / "df.pkl"
    test_data_path = BASE_DIR / "data" / "test_df.pkl"
    
    # 1. Pipeline execution (Only run if data doesn't exist)
    if not os.path.exists(data_path):
        prepare_data(data_path)
    else:
        print("Dataset already exists. Loading from disk...")
        
    df = pd.read_pickle(data_path)
    print(f"Data shape: {df.shape}")
    
    # 2. Train Model
    model = train_logistic_model(df, epochs=200)
    
    # 3. Process Hand-Crafted Test Data
    print("\n--- Evaluating on Hand-Crafted Test Data ---")
    test_df = prepare_handcrafted_test_data(test_data_path)
    
    # Convert DataFrame columns to PyTorch tensors
    X_test_handcrafted = torch.tensor(np.stack(test_df['embedding'].values), dtype=torch.float32)
    y_test_handcrafted = torch.tensor(test_df['sentiment'].values, dtype=torch.float32).unsqueeze(1)
    
    # 4. Predict and Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_handcrafted) # These are the exact confidence values (probabilities)
        test_preds = (test_outputs > 0.5).float()
        
        # Calculate overall accuracy
        correct = (test_preds == y_test_handcrafted).sum().item()
        total = y_test_handcrafted.size(0)
        accuracy = correct / total
        
        print(f"\nHand-Crafted Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)\n")
        print("--- Detailed Breakdown of Misclassified Statements ---")
        
        # Convert tensors to numpy arrays for easier iteration
        probs = test_outputs.squeeze().numpy()
        preds = test_preds.squeeze().numpy()
        actuals = y_test_handcrafted.squeeze().numpy()
        statements = test_df['statement'].values
        
        errors_found = 0
        
        # Loop through and print ONLY the incorrect results with confidence
        for i in range(len(statements)):
            if preds[i] != actuals[i]:
                errors_found += 1
                sentiment_str = "Positive" if preds[i] == 1 else "Negative"
                actual_str = "Positive" if actuals[i] == 1 else "Negative"
                
                print(f"Statement:  {statements[i]}")
                print(f"Prediction: {sentiment_str} (Actual: {actual_str})")
                print(f"Confidence: {probs[i]:.4f}")
                print("-" * 60)
                
        if errors_found == 0:
            print("None! All statements were classified correctly.")

if __name__ == "__main__":
    main()