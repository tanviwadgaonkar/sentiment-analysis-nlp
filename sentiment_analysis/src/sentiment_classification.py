import pickle
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

# Load the model and vectorizer
with open('../model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('../vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Fetch preprocessed tweets without labels
cursor.execute("SELECT id, preprocessed_text FROM tweets WHERE sentiment IS NULL")
tweets = cursor.fetchall()

# Classify tweets
for tweet_id, text in tweets:
    text_vec = vectorizer.transform([text])
    sentiment = model.predict(text_vec)[0]
    cursor.execute("""
        UPDATE tweets
        SET sentiment = %s
        WHERE id = %s
    """, (sentiment, tweet_id))
    db.commit()
