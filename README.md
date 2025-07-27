# Flight Ranking Competition - Experimental Neural Network


[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://github.com/goonzard/flight-ranking-competition/blob/main/notebooks/001.eda.ipynb)
[![pytorch](https://img.shields.io/badge/pytorch-1.9.1-EE4C2C.svg)](https://pytorch.org/)
[![kaggle](https://img.shields.io/badge/kaggle-Flight%20Ranking%20Competition-00B8D9.svg)](https://www.kaggle.com/c/flight-ranking-competition)
[![python](https://img.shields.io/badge/python-3.8.5-3776AB.svg)](https://www.python.org/)


## 📌 Overview

This project tackles a group‑wise ranking problem on a large flight search dataset.
Given a ranker_id (a search session) and multiple flight candidates, the goal is to rank the options so that the chosen flight appears as high as possible in the ranking.
The evaluation metric is HitRate@3 – the probability that the chosen flight appears in the top‑3 predicted scores.

## 📂 Data Engineering Pipeline

### The raw dataset consists of:

* Flight information: textual descriptions (initially embedded), numerical features, and boolean/categorical flags.
  * ***EXPERIMENTAL STEP: Create textual descriptions from flight features and turned it into embeddings using ontology embeddings steps***
* User information: numerical features (e.g., profile stats), categorical/boolean attributes.
* Pricing information: numerical and categorical/boolean indicators.

### Steps performed:

1. Parquet Data Processing
    * Raw data split into multiple Parquet files for streaming.
    * Used Polars to read, merge, and process large files efficiently.
    * Excluded non‑feature columns: row_id, ranker_id, and selected.
2. Normalization
    * Normalization statistics computed from an initial training split.
    * All features scaled consistently across train/validation/test.
3. Group‑wise DataLoader
    * Implemented a custom ParquetRankDataset class.
    * Ensures each batch contains complete groups (ranker_id) to calculate ranking losses correctly.
    * Streams data in manageable chunks (max_rows parameter).
4. Feature Engineering
    * Initial experiments included text embeddings for flight descriptions (OpenAI embeddings reduced to 480‑dim vectors).
    * Final solution switched to pure tabular features without embeddings for stability and generalization.

## 🧠 Models Tried

### ✅ 1. Neural Network Ranker (MLP)

Architecture:
Example: 480 → 512 → 256 → 128 → 1
* BatchNorm, GELU activations, and Dropout (0.2) for regularization.
* Trained with gradient clipping and AdamW optimizer.
* Pros:
  * High flexibility, can learn complex interactions.
* Cons:
  * Tended to overfit with noisy embeddings.
  * Training time and tuning complexity.

### ⚙️ Training and Validation
* Validation: Performed on held‑out splits (e.g., fold 8 & 9).
* Metrics logged: LambdaLoss, HitRate@3.
* Early stopping: Based on validation LambdaLoss with patience.
* Scheduler: Cosine Annealing learning rate used in neural net phase.


## 📊 Result

* Not good at all. Change to Boosting without NLP.