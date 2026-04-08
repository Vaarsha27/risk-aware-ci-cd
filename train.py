import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from preprocess import preprocess_data

def train_model():
    X, y = preprocess_data("data/augmented_dataset.csv")

    # ✅ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            verbosity=0
        )
    )

    model.fit(X_train, y_train)

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print("✅ Model trained with train-test split!")

if __name__ == "__main__":
    train_model()