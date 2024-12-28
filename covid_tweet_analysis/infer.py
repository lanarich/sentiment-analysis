import lightgbm as lgb
import numpy as np

from .constants import MODELS_PATH
from .embeddings_model import get_embeddings


def infer(text: str, model_name: str):
    model = lgb.Booster(model_file=f"{MODELS_PATH}/{model_name}")
    embedding = get_embeddings(text, embedding_dim=1024)
    prediction = model.predict([embedding])
    if isinstance(prediction, (list, np.ndarray)):
        prediction = prediction[0]
    prediction_class = 1 if prediction[0] > 0.5 else 0
    return prediction_class
