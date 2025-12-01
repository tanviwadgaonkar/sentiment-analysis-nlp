# Sentiment Analysis on Twitter Data

A machine learning pipeline that classifies tweet sentiment (positive, negative, neutral) using NLP techniques and classical ML models.

## Features
- Text cleaning (lowercasing, stopword removal, punctuation removal)
- Tokenization and TF-IDF feature extraction
- ML model training and evaluation
- Sentiment distribution visualization

## Tech Stack
- Python  
- NumPy, Pandas  
- scikit-learn  
- NLTK (or similar)  
- Matplotlib

## How to Run

```bash
git clone https://github.com/tanviwadgaonkar/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

pip install -r requirements.txt

# Example commands
python preprocess.py
python train.py
python evaluate.py
