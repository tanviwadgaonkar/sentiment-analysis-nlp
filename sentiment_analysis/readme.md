# Sentiment Analysis Project

This project involves developing a sentiment analysis model for Twitter data. The workflow includes collecting tweets, preprocessing the text data, training a sentiment classification model, classifying the sentiments of all tweets, and visualizing the results using Tableau.

## Project Structure

- `data/`: Directory for storing data files.
- `database/`: Directory for database-related scripts.
- `src/`: Directory for source code.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Setup

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Set up MySQL database and update `config.json` with your credentials.

3. Run the scripts in the following order:
    - `data_collection.py`: Collects tweets and stores them in the database.
    - `preprocessing.py`: Preprocesses the text data.
    - `model_training.py`: Trains the sentiment analysis model.
    - `sentiment_classification.py`: Classifies the sentiment of all tweets.
    - `export_for_tableau.py`: Exports the data for visualization in Tableau.

## Configuration

Update `config.json` with your Twitter API and MySQL database credentials.
