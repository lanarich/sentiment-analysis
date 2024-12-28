import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .constants import MODELS_PATH, TEST_DATA_PATH
from .embeddings_model import add_embeddings_to_df


def test_model():
    test_data = pd.read_csv(TEST_DATA_PATH, encoding="utf-8", encoding_errors="replace")
    test_data_emd = add_embeddings_to_df(
        test_data, text_column="OriginalTweet", embedding_dim=1024
    )
    X_test = test_data_emd.drop(
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
    y_test = test_data_emd["Sentiment"]
    model = lgb.Booster(model_file=f"{MODELS_PATH}/model.txt")
    y_pred = model.predict(X_test)
    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
