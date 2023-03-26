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
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
url = "Amazon_reviews_plus_LLM.csv"
data = pd.read_csv(url)
data = data.fillna('')

# Preprocessing
stop_words = set(stopwords.words('english'))
data['reviewText'] = data['reviewText'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words])) # Remove stop words
data['reviewText'] = data['reviewText'].apply(lambda x: ' '.join([word for word in x.lower().split() if word.isalpha()])) # Remove non-alphabetic characters

# Create feature matrix and target vector
X = data['reviewText']
y = np.where(data['llm'] == True, 1, 0)

# Vectorize text using Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Train and evaluate model using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', cm)
