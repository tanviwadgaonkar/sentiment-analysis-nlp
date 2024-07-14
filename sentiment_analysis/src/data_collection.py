import tweepy
import mysql.connector
from datetime import datetime
import json

# Load configuration from file
with open('../config.json') as config_file:
    config = json.load(config_file)

# Twitter API credentials
consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

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

# Create table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS tweets (
        id VARCHAR(255) PRIMARY KEY,
        text TEXT,
        created_at DATETIME
    )
""")

# Function to store tweets in the database
def store_tweet(id, text, created_at):
    cursor.execute("""
        INSERT INTO tweets (id, text, created_at)
        VALUES (%s, %s, %s)
    """, (id, text, created_at))
    db.commit()

# Twitter API authentication
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Collect tweets
for tweet in tweepy.Cursor(api.search, q='your_search_query', lang='en', since='2023-01-01').items(1000):
    store_tweet(tweet.id_str, tweet.text, tweet.created_at)
