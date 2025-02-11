import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save TF-IDF features
import scipy
scipy.sparse.save_npz('X_train_tfidf.npz', X_train_tfidf)
scipy.sparse.save_npz('X_test_tfidf.npz', X_test_tfidf)

# Save the vectorizer
import pickle
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)