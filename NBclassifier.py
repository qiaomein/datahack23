# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:22:14 2023

@author: Mia_V
"""

import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

text = "reviewText"
# Load the dataset
url = "Amazon_reviews_plus_LLM.csv"
data = pd.read_csv(url)

data = data.fillna('')

# Preprocessing
stop_words = set(stopwords.words('english'))
data[text] = data[text].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words])) # Remove stop words
data[text] = data[text].apply(lambda x: ' '.join([word for word in x.lower().split() if word.isalpha()])) # Remove non-alphabetic characters

# Create feature matrix and target vector
X = data[text]
y = np.where(data['llm'] == True, 1, 0)

# Vectorize text using Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.98, random_state=32)

# Train and evaluate model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', cm)