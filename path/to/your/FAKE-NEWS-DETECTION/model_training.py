import pandas as pd
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load TF-IDF features and labels
X_train_tfidf = scipy.sparse.load_npz('X_train_tfidf.npz')
X_test_tfidf = scipy.sparse.load_npz('X_test_tfidf.npz')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)