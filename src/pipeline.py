# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:51:46 2026

@author: patri

Loads the Twitter data, creates the text embedding and saves all
results into a DataFrame
"""

# src/pipeline.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from nltk.corpus import twitter_samples
import pandas as pd 

def prepare_data(data_path):
    print("Preparing data and fetching embeddings from OpenAI...")
    load_dotenv()
    
    # Load Tweets
    tweets_pos = twitter_samples.strings('positive_tweets.json')
    tweets_neg = twitter_samples.strings('negative_tweets.json')
    tweets = tweets_pos + tweets_neg

    # Sentiment labels
    labels = [1] * len(tweets_pos) + [0] * len(tweets_neg)

    # Embed tweets
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

    batch_size = 1000  
    all_embeddings = []

    for i in range(0, len(tweets), batch_size):
        batch = tweets[i : i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
            dimensions=256
        )
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)

    # Assemble and Save DataFrame
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df = pd.DataFrame({"tweet": tweets, "sentiment": labels, "embedding": all_embeddings})
    df.to_pickle(data_path)
    print(f"Data saved to {data_path}")