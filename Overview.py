"""
=====================================
Sentiment Analysis Project Overview
=====================================

Goal:
-----
We aim to classify 10_000 tweets (5_000 positive, 5_000 negative) using three different
machine learning approaches of increasing complexity:

    1. Fixed embeddings + Logistic Regression
    2. Fixed embeddings + Small Neural Network (MLP)
    3. Fixed embeddings + Transformer

---------------------------------------------------------------------
DATA PIPELINE (COMMON TO ALL APPROACHES)
---------------------------------------------------------------------

We start by converting each tweet into a numerical representation using:

    OpenAI embedding model: "text-embedding-3-small"

Properties:
    - Embedding dimension: 256
    - Each tweet → vector x ∈ ℝ^256
    - These embeddings capture semantic meaning of the text

Result:
    Dataset D = {(x_i, y_i)}, where:
        x_i ∈ ℝ^256
        y_i ∈ {0,1} (negative, positive)

---------------------------------------------------------------------
APPROACH 1: LOGISTIC REGRESSION (LINEAR BASELINE)
---------------------------------------------------------------------

Model:
    z = w^T x + b
    p(y=1|x) = sigmoid(z)

Parameters:
    - Weight vector: w ∈ ℝ^256
    - Bias: scalar
    - Total parameters: 257

Training:
    - Loss: Binary Cross Entropy (log-likelihood)
    - Optimization: Gradient Descent / LBFGS
    - Regularization: L2 

Interpretation:
    - Each embedding dimension is treated as a feature
    - The model learns a linear decision boundary in embedding space

Advantages:
    - (Hopefully) very fast
    - Works well with small datasets
    - Strong baseline

Limitations:
    - Only captures linear relationships

---------------------------------------------------------------------
APPROACH 2: SMALL NEURAL NETWORK (MLP)
---------------------------------------------------------------------

Model:
    x (256)
      ↓
    Linear(256 → 128)
      ↓
    ReLU
      ↓
    Linear(128 → 1)
      ↓
    Sigmoid

Architecture:
    - Input layer: 256
    - Hidden layer: 128 units
    - Output: 1 (binary classification)

Parameters:
    - ~30K parameters (approx)

Training:
    - Loss: Binary Cross Entropy
    - Optimizer: Adam (lr ~ 1e-3)

Advantages:
    - Captures non-linear relationships
    - Still lightweight and fast

Limitations:
    - Slightly higher risk of overfitting
    - Still depends on fixed embeddings

---------------------------------------------------------------------
APPROACH 3: TRANSFORMER 
---------------------------------------------------------------------

Goal:
    classifier jointly from raw text.

Model Architecture:

    Transformer encoder:
        - Number of layers: 2
        - Number of attention heads: 4
        - Hidden dimension: 128
        - Feedforward dimension: 256

    Classification head:
        - Mean pooling or [CLS] token
        - Linear(128 → 1)
        - Sigmoid

Training:
    - Loss: Binary Cross Entropy
    - Optimizer: Adam (lr ~ 2e-4)

Advantages:
    - More flexible and expressive

Limitations:
    - Harder to train with small data (4000 samples)
    - Likely underperforms pretrained models
    - Requires more tuning

---------------------------------------------------------------------
RECOMMENDED EXPERIMENTAL FLOW
---------------------------------------------------------------------

1. Start with Logistic Regression
    → Establish baseline accuracy (~75–85%)

2. Train MLP
    → Expect modest improvement (~+2–5%)

3. Train Transformer
    → Likely similar or slightly worse unless tuned carefully

---------------------------------------------------------------------
PRACTICAL NOTES
---------------------------------------------------------------------

- Always split data:
    Train: 70%
    Validation: 15%
    Test: 15%

- Normalize embeddings (optional but helpful)

- Use early stopping for neural models

- Evaluation metrics:
    - Accuracy
    - F1-score (important if imbalance occurs)

---------------------------------------------------------------------
SUMMARY
---------------------------------------------------------------------


Approach        | Learns Embeddings | Complexity | Expected Performance
----------------|------------------|------------|---------------------
Logistic Reg.   | No               | Low        | Strong baseline
MLP             | No               | Medium     | Slight improvement
Transformer     | No              | High       | Educational, harder

---------------------------------------------------------------------
CONCLUSION
---------------------------------------------------------------------

This project demonstrates three key paradigms in NLP:

    1. Fixed feature + linear model
    2. Fixed feature + nonlinear model
    3. End-to-end representation learning

For a small dataset like ours, simpler models are often surprisingly competitive.

"""