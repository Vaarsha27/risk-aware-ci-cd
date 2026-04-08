from sklearn.metrics import classification_report
from preprocess import preprocess_data
import joblib

def evaluate():
    X, y = preprocess_data("data/augmented_dataset.csv")

    model = joblib.load("model/model.pkl")

    y_pred = model.predict(X)

    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate()