import os
import re
import string
import joblib
import argparse
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score


nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def train_model(csv_path):

    if not os.path.exists(csv_path):
        print(" CSV file not found!")
        return

    print(" Loading dataset...")
    df = pd.read_csv(csv_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    print(" Cleaning text...")
    df['cleaned'] = df['message'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['label'], test_size=0.2, random_state=42
    )

    print(" Building pipeline...")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression())
    ])

    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['liblinear']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

    print(" Training model...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\n Best Parameters:", grid.best_params_)

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    print("\n Evaluation Results")
    print("Accuracy:", accuracy)
    print("ROC AUC:", roc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cv_score = cross_val_score(best_model, df['cleaned'], df['label'], cv=5).mean()
    print("Cross Validation Accuracy:", cv_score)

    print("\n Saving model...")
    joblib.dump(best_model, "advanced_spam_classifier.pkl")

    print(" Training complete! Model saved as advanced_spam_classifier.pkl")

def predict_message(message):
    if not os.path.exists("advanced_spam_classifier.pkl"):
        print(" Model not found. Train first.")
        return

    model = joblib.load("advanced_spam_classifier.pkl")
    cleaned = clean_text(message)
    prediction = model.predict([cleaned])[0]

    if prediction == 1:
        print(" SPAM")
    else:
        print(" HAM (Not Spam)")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Path to spam.csv file")
    parser.add_argument("--predict", help="Predict a single message")

    args = parser.parse_args()

    if args.train:
        train_model(args.train)

    elif args.predict:
        predict_message(args.predict)

    else:
        print("Usage:")
        print("Train model:")
        print("python advanced_spam_classifier.py --train spam.csv")
        print("\nPredict message:")
        print('python advanced_spam_classifier.py --predict "Free entry in 2 a wkly comp!"')