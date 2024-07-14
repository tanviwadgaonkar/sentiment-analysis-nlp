import re
import nltk
import mysql.connector
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import json

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

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

# Fetch tweets
cursor.execute("SELECT id, text FROM tweets")
tweets = cursor.fetchall()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stop words
    tokens = [PorterStemmer().stem(word) for word in tokens]  # Stemming
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Store preprocessed tweets
for tweet_id, text in tweets:
    preprocessed_text = preprocess_text(text)
    cursor.execute("""
        UPDATE tweets
        SET preprocessed_text = %s
        WHERE id = %s
    """, (preprocessed_text, tweet_id))
    db.commit()
