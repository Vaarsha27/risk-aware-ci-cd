import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Numerical features (fixed order)
    num_cols = [
        'error_count',
        'duration',
        'test_failed',
        'coverage',
        'cpu_usage',
        'memory_usage',
        'past_failures'
    ]

    # Text feature
    text_col = 'log_text'

    # Target columns
    target_cols = [
        'Failure_Risk',
        'Performance_Risk',
        'Test_Risk',
        'Security_Risk'
    ]

    # Fill missing values
    df[num_cols] = df[num_cols].fillna(0)
    df[text_col] = df[text_col].fillna("")

    # Scale numerical
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_text = tfidf.fit_transform(df[text_col]).toarray()

    # Combine
    X = np.hstack((X_num, X_text))
    y = df[target_cols]

    # Save preprocessors
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(tfidf, "model/tfidf.pkl")

    return X, y