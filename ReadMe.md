# Financial Sentiment Analysis: A Transfer Learning Approach via LLM Embeddings

An end-to-end machine learning pipeline that leverages **OpenAI text embeddings** and **Transfer Learning** to accurately classify the sentiment of complex, domain-specific financial text. 

Built with PyTorch, this project demonstrates how to achieve LLM-level analytical accuracy at a fraction of the compute time and API cost.

## Executive Summary

* **Transfer Learning:** The model is trained on a generalized dataset of 10,000 tweets and evaluated zero-shot on highly complex, domain-specific financial statements and macroeconomic paragraphs. 
* **Ultra-Fast & Cost-Effective:** Instead of querying an LLM (like GPT-5) for every prediction, which is slow and expensive, this pipeline fetches `text-embedding-3-small` vectors (256 dimensions) and processes them through a lightweight PyTorch Logistic Regression model. 
* **High Accuracy:** Achieved **94.60% Validation Accuracy** on the training domain and an impressive **94.12% Accuracy** on the handcrafted financial test set.

## Why This Approach? (Architecture Rationale)

In modern NLP, sending millions of rows of text to an LLM API for sentiment classification is financially and computationally prohibitive. 

By separating the **Semantic Extraction** (OpenAI Text Embeddings) from the **Classification** (PyTorch Logistic Regression), this architecture provides:
1. **Extremely Cheap Inference:** Embedding vectors cost a fraction of a cent per batch compared to generative LLM tokens. Unlike generative models that require an expensive, iterative forward pass for every single token they generate (auto-regressive decoding), embedding models require only a single, non-iterative pass to output a fixed-size vector. This makes the compute cost exponentially lower.
2. **Lightning-Fast Training:** Training a logistic regression model on 256-dimensional vectors takes seconds on a CPU, establishing a rapid feedback loop.
3. **Rich Semantic Understanding:** The foundational embeddings already understand human language, allowing a simple linear decision boundary to solve a complex classification problem.

## Under the Hood: The Mechanics of the Model

To build an efficient pipeline, the model relies on two core mathematical concepts: dense vector representation and iterative optimization.

### 1. How the Semantic Embeddings Work
Instead of using brittle keyword-matching techniques (like Bag of Words or TF-IDF), this project maps raw text into a continuous mathematical space using OpenAI's `text-embedding-3-small` model. Every document is compressed into a dense, **256-dimensional vector** ($x \in \mathbb{R}^{256}$). In this vector space, semantically similar sentences are positioned closer together. Because the OpenAI model has already been pre-trained on vast amounts of data, these vectors inherently capture context, sentiment, and nuance, effectively doing the "heavy lifting" of language comprehension before our classifier even sees the data.

### 2. Training the Logistic Regression via Gradient Descent
The classifier is a PyTorch-based Logistic Regression model that learns a linear decision boundary in this 256-dimensional embedding space. The model computes a weighted sum of the input vector ($z = w^T x + b$) and passes it through a Sigmoid activation function to output a probability between 0 and 1. 

To train the model, **Gradient Descent** (specifically the Adam optimizer) is used. With just 200 epochs the model has learned which semantic features in the embedding space strongly correlate with positive or negative sentiment.

## The Dataset & Evaluation

To prove the model didn't just learn trivial sentiment (e.g., "The stock is good"), the evaluation is run on a **challenging test dataset** containing sophisticated financial jargon. The full set of examples is in `src/test_data.py`. 

**Examples of correctly classified complex statements:**
> *"The Treasury auction saw a record 2.8x bid-to-cover ratio with heavy participation from indirect bidders."* (Positive)

> *"The company’s 'Days Sales Outstanding' (DSO) increased by 12 days despite achieving record top-line revenue."* (Negative)

> *"Free cash flow came in slightly below expectations, driven in part by temporary working capital pressures; nonetheless, the company reduced its debt levels and announced a modest share repurchase program. Despite some mixed underlying metrics, investors appeared reassured by management’s confidence in medium-term growth prospects and its disciplined approach to capital allocation."* (Positive)

> *"The company announced a new strategic initiative aimed at revitalizing growth, which was initially welcomed by investors; nevertheless, its latest results revealed slowing momentum across core segments and a continued deterioration in cash flow. Management acknowledged the challenges but offered few concrete details on how or when conditions might improve, adding to market uncertainty."* (Negative)


## Performance & Results

The model converged smoothly over 200 epochs using the Adam optimizer and Binary Cross Entropy loss.

* **Training Size:** 7,000 samples of tweets (not containing any financial context.)
* **Validation Accuracy:** 94.60%
* **Financial Test Accuracy:** **94.12% (64/68 correct)**

### Error Analysis
The model only misclassified 4 out of 68 statements. A closer look reveals that it struggled exclusively with highly nuanced financial mechanics where the semantic wording contradicts standard sentiment rules:
* *Misclassification:* "The yield curve inversion deepened today..." (Predicted Positive, Actual Negative). *Reason:* "Deepened" is often semantically positive in standard English, but negative in bond markets.
* *Misclassification:* "The merger arbitrage spread narrowed to 1.5%..." (Predicted Negative, Actual Positive). 

*Takeaway:* The model demonstrates a profound understanding of general and business sentiment, only failing on niche Wall Street mechanics that would typically require a specialized financial LLM (like FinBERT) to parse.

## Tech Stack & Structure

* **Language:** Python
* **Deep Learning:** PyTorch (`torch`, `torch.nn`)
* **Data Processing:** Pandas, NumPy, NLTK
* **APIs:** OpenAI (`text-embedding-3-small`)

```text
├── data/                   # Pickled DataFrames containing text and embeddings
├── src/
│   ├── pipeline.py         # Data fetching and OpenAI API embedding generation
│   ├── model_logistic.py   # PyTorch Logistic Regression architecture and training loop
│   ├── test_data.py        # Hand-crafted complex financial test dataset
│   └── utils.py            # Train/Val/Test splitting logic
├── main.py                 # Main execution script
├── .env                    # Environment variables (OpenAI Key)
