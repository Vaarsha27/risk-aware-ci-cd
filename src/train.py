import joblib
import os

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from preprocess import preprocess_data

def train_model():
    # Load data
    X, y = preprocess_data("data/augmented_dataset.csv")

    # Model
    model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    )

    # Train
    model.fit(X, y)

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print("Model trained & saved!")

if __name__ == "__main__":
    train_model()