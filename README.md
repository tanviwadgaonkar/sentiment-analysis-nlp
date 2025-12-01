# Sentiment Analysis on Twitter Data

A machine learning pipeline that classifies tweet sentiment (positive, negative, neutral) using NLP techniques and classical ML models.

## Features
- Collects and stores tweet data  
- Text cleaning (lowercasing, stopword removal, punctuation removal)  
- Tokenization and TF-IDF feature extraction  
- Model training and evaluation  
- Export of results for visualization (Tableau)

## Tech Stack
- Python  
- NumPy, Pandas  
- scikit-learn  
- NLTK  
- Matplotlib

## Folder Structure

sentiment_analysis/

├── data/ # Tweet dataset

├── database/ # Database files

├── src/

│ ├── data_collection.py

│ ├── export_for_tableau.py

│ ├── model_training.py

│ ├── preprocessing.py

│ └── sentiment_classification.py

# How to Run

1. Preprocess the data

python src/preprocessing.py
 
 2. collect data

python src/data_collection.py
 
 3. Train the model

python src/model_training.py
 
 4. Run sentiment classification

python src/sentiment_classification.py

 5. Export results for Tableau dashboard

python src/export_for_tableau.py

 # My Contribution

Implemented preprocessing pipeline

Trained ML models and evaluated performance

Built sentiment classification script

Exported results to Tableau for visualization and analysis
