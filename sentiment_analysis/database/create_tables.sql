-- Create the database if it does not exist
CREATE DATABASE IF NOT EXISTS sentiment_analysis;

-- Use the database
USE sentiment_analysis;

-- Create the table for storing tweets
CREATE TABLE IF NOT EXISTS tweets (
    id VARCHAR(255) PRIMARY KEY,            -- Tweet ID
    text TEXT,                              -- Original tweet text
    preprocessed_text TEXT,                 -- Preprocessed tweet text
    sentiment VARCHAR(50),                  -- Sentiment classification
    created_at DATETIME                     -- Date and time when the tweet was created
);

