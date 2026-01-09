## Project: Receipt line-item classifier
DTU course: 02476, machine learning operations

### Goal
Build an end-to-end ML system that classifies short transaction descriptions (receipt line-items / bank statement text) into spending categories (e.g., Food & Dining, Transportation, Utilities, Shopping). Input is a single text string, output is a predicted category + confidence. The project emphasizes a reproducible MLOps pipeline: data download + preprocessing, training + evaluation, experiment tracking, packaging and deployment of an inference API, and automated testing/CI.

### Dataset
We will use a public labeled transaction dataset from Hugging Face. Minimum required columns:
- `transaction_description` (string): short free-text description
- `category` (string): target label (one of the predefined categories)

If available, we may also use: `amount`, `currency`, `country`, `timestamp` / derived time features.

The full dataset is large (millions of rows), so we will start with a small subset to iterate quickly. Plan:
- Subsample to 50k–200k rows for development (stratified by `category`)
- Train/val/test split (e.g., 80/10/10)
- Basic text cleaning (trim, normalize whitespace, optional lowercasing)
- Handle rare categories by merging into `Other` or filtering if needed

Link to dataset: https://huggingface.co/datasets/sreesharvesh/transactiq-enriched

### Expected models
1. **Baseline:** TF-IDF + linear classifier (Logistic Regression / Linear SVM)
2. **Neural:** Fine-tuned Transformer (e.g., DistilBERT) for multi-class classification

We will compare models using accuracy and macro-F1 and deploy the best-performing/most robust model behind a simple API that returns category + confidence.
