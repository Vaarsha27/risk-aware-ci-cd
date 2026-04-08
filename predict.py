import numpy as np
import joblib

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
tfidf = joblib.load("model/tfidf.pkl")

def predict_risk(input_num, input_text):
    num_scaled = scaler.transform([input_num])
    text_vec = tfidf.transform([input_text]).toarray()

    final_input = np.hstack((num_scaled, text_vec))

    prediction = model.predict(final_input)

    return prediction