# Data Lineage

## Train

Test follows approximately the same lineage

1. Source data
  * ./kaggle/train.parquest

2. External Data (public)
  * ./kaggle/support
  * ./notebooks/002.ontology-airline.ipynb

3. Engineering - 1
  * Notebooks
    * `./notebooks/002.data-engineering-flight.ipynb`: Using external data, create natural language flight information data
    * `./notebooks/002.data-engineering-price.ipynb`: Extract price information from source data
    * `./notebooks/002.data-engineering-user.ipynb`: Extract demographic information from source data
    * `./notebooks/002.data-engineering-additional.ipynb`: Extract group wise price difference, is preferred flight etc
  * Resulting files (respectively)
    * `./data/processed_flight_features_train.parquet`
    * `./data/processed_pricing_features_train.parquet`
    * `./data/processed_user_features_train.parquet`
    * `./data/processed_additional_features_train.parquet`
4. Engineering - 2. (Embedding)
  * Primarily use NLP data
  * Uses MiniLM model to create embeddings (384 dimensions)
  * Notebooks
    * `./notebooks/002.data-engineering-flight-embed-train-lite.ipynb`: Create embeddings for flight data
  * Resulting files
    * `./data/embed_flight_feature_train.parquet`
5. Split
  * For memory reasons, split data into 15 parts and feed them onto neural network model by streaming
  * `./data/train/train_split_0.parquet`