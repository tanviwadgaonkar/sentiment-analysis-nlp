import pandas as pd
import mysql.connector
import json

# Load configuration from file
with open('../config.json') as config_file:
    config = json.load(config_file)

# MySQL configuration
db_config = config['mysql']

# Connect to MySQL database
db = mysql.connector.connect(
    host=db_config['host'],
    user=db_config['user'],
    password=db_config['password'],
    database=db_config['database']
)
cursor = db.cursor()

# Fetch all tweets with sentiment
cursor.execute("SELECT * FROM tweets")
data = cursor.fetchall()

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data, columns=['id', 'text', 'created_at', 'preprocessed_text', 'sentiment'])
df.to_csv('../data/tweets_with_sentiment.csv', index=False)
