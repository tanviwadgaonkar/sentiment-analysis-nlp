import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import mysql.connector
import json
import pickle

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

# Fetch preprocessed tweets with labels (assuming labels are stored in the database)
cursor.execute("SELECT preprocessed_text, sentiment FROM tweets WHERE sentiment IS NOT NULL")
data = cursor.fetchall()

# Convert to DataFrame
df = pd.DataFrame(data, columns=['text', 'sentiment'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Model evaluation
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open('../model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('../vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
