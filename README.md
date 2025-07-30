# ✈️ Flight Ranking Competition

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://github.com/goonzard/flight-ranking-competition/blob/main/notebooks/001.eda.ipynb)
[![pytorch](https://img.shields.io/badge/pytorch-1.9.1-EE4C2C.svg)](https://pytorch.org/)
[![kaggle](https://img.shields.io/badge/kaggle-Flight%20Ranking%20Competition-00B8D9.svg)](https://www.kaggle.com/c/flight-ranking-competition)
[![python](https://img.shields.io/badge/python-3.12.0-3776AB.svg)](https://www.python.org/)

---

## 📌 Overview

This project tackles a **group‑wise ranking problem** on a large flight search dataset.  
Given a `ranker_id` (i.e. a search session) and multiple flight candidates, the goal is to rank the options so that the chosen flight appears as high as possible.

**Evaluation Metric**: `HitRate@3` – probability that the selected flight appears in the top-3 predictions.

---

## 🛠️ Tech Stack

- `Polars` for fast tabular processing
- `Pytorch` for deep ranking models
- `XGBoost`, `LightGBM` for boosted tree ranking
- `Optuna` for hyperparameter optimization

---

## 📂 Data Engineering Pipeline

### 🔸 Dataset Overview

- **Flight information**:  
  Initially processed into textual embeddings (OpenAI), then reduced to tabular form (numerical/categorical).
- **User features**:  
  Profile-level attributes, mainly numeric/categorical.
- **Price features**:  
  Numerical prices, tax breakdowns, cabin classes, baggage rules, etc.

### 🔸 Pipeline Steps

1. **Parquet Streaming via Polars**
   - Efficient multi-file merging for large datasets
   - Dropped non-feature columns (`row_id`, `ranker_id`, `selected`)

2. **Normalization**
   - Stats computed on training split
   - Applied consistently across train/validation/test

3. **Group-wise Streaming Loader**
   - Custom `ParquetRankDataset`
   - Preserves `ranker_id` boundaries in every batch
   - Streaming enabled via `max_rows` setting

4. **Feature Engineering**
   - ❌ Initial: Text embeddings (OpenAI, 480‑dim) – later dropped
   - ✅ Final: Clean tabular features only — numeric + boolean + categorical

---

## 🧠 Models Tried

### ✅ Neural Network Ranker (MLP)

> **Used early in experimentation — ultimately dropped**

- **Arch**: `480 → 512 → 256 → 128 → 1`  
- **Techs**: BatchNorm, GELU, Dropout(0.2), AdamW, GradClip  
- **Pros**: Expressive capacity for nonlinearity  
- **Cons**: Overfit quickly, slow tuning, large memory footprint

### ✅ Boosted Tree Rankers (Final Solution)

- **XGBoost**: Best `NDCG@3 ≈ 0.41`
- **LightGBM**: Best `NDCG@3 ≈ 0.42` (winning model)

---

## ⚙️ Training & Validation

- Train/Val split by `ranker_id`
- Rank-aware loss (`LambdaLoss`, `NDCG`)
- Early stopping on validation metric
- Used `CosineAnnealingLR` in NN training phase
- Optuna used for boosting model hyperparameter tuning

---

## 🧠 NLP Feature Engineering: Lessons Learned

During early experimentation, I used NLP embeddings (OpenAI) to encode textual flight descriptions (routes, baggage rules, fare types). However, this approach was later discarded due to poor performance in ranking tasks.

### ⚠️ NLP Embeddings Are Not Suitable for Precise Quantitative Ranking

While powerful for semantic similarity, embeddings lack fine-grained numerical precision needed for ranking.

#### 🔍 Key Issues:
- **Loss of precision**: Small value changes (e.g., 1h vs 8h layover) are smoothed away
- **Unstable behavior**: Tiny wording changes shift the entire embedding
- **Opaque structure**: No direct mapping between vector space and key flight properties

---

### ✅ When NLP Works Well:
- Classification (e.g., LCC vs Full-service airlines)
- Semantic clustering / retrieval
- Keyword flagging (e.g., "non-refundable", "no cabin baggage")

### ❌ When NLP Struggles:
- Ranking models (`NDCG`, `HitRate`)
- Regression (e.g., predicting price or duration)
- Tasks requiring stability, interpretability, or numeric consistency

---

### 💡 Takeaway

> NLP embeddings are excellent for *soft classification and clustering*,  
> but unreliable for *precise quantitative modeling* in ranking or regression.

For future work, structured extraction (regex parsing, token flags, keyword presence) is preferred over full-sentence embeddings.

---

## 📊 Final Result Summary

| Model      | Score (NDCG@3) | Notes                         |
|------------|----------------|-------------------------------|
| XGBoost    | 0.41           | Stable, moderate performance  |
| LightGBM   | 0.42           | **Best model**                |
| NN + NLP   | ~0.33          | Overfit, noisy, unstable      |

---

Let me know if you'd like to generate a performance chart, include feature importance plots, or share submission templates.