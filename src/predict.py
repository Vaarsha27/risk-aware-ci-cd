import numpy as np
import joblib

# Load saved components
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
tfidf = joblib.load("model/tfidf.pkl")

def predict_risk(input_num, input_text):
    # Scale numerical
    num_scaled = scaler.transform([input_num])

    # Transform text
    text_vec = tfidf.transform([input_text]).toarray()

    # Combine
    final_input = np.hstack((num_scaled, text_vec))

    # Predict
    prediction = model.predict(final_input)

    return prediction