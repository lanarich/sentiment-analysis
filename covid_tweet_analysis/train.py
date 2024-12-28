import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .constants import MODELS_PATH, RANDOM_STATE, TRAIN_DATA_PATH, VAL_SIZE
from .embeddings_model import add_embeddings_to_df
from .test import test_model


def train():
    # Load and preprocess data
    data = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8", encoding_errors="replace")
    data_emd = add_embeddings_to_df(
        data, text_column="OriginalTweet", embedding_dim=1024
    )
    X = data_emd.drop(
        [
            "Sentiment",
            "UserName",
            "ScreenName",
            "Location",
            "TweetAt",
            "Unnamed: 0",
            "OriginalTweet",
        ],
        axis=1,
    )
    y = data_emd["Sentiment"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)

    validate_model(model, X_val, y_val)

    model.booster_.save_model(f"{MODELS_PATH}/model.txt")

    test_model()


def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Model Accuracy: {accuracy}")
