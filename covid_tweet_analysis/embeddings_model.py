import numpy as np
import requests

from .constants import BASE_EMBEDDING_URL, EMBEDDING_MODEL_NAME


def get_embeddings(text: str, embedding_dim: int = 512):
    """
    Получает эмбеддинги для текста через API и фиксирует размерность.
    """
    response = requests.post(
        BASE_EMBEDDING_URL, json={"model": EMBEDDING_MODEL_NAME, "input": [text]}
    )
    embedding = np.array(response.json()["data"][0]["embedding"])

    if embedding.shape[0] > embedding_dim:
        embedding = embedding[:embedding_dim]
    elif embedding.shape[0] < embedding_dim:
        embedding = np.pad(
            embedding, (0, embedding_dim - embedding.shape[0]), mode="constant"
        )

    return embedding


def add_embeddings_to_df(df, text_column, embedding_dim: int = 512):

    """
    Добавляет эмбеддинги в датафрейм как отдельные столбцы.
    """

    embeddings = df[text_column].apply(lambda x: get_embeddings(x, embedding_dim))
    embeddings_array = np.stack(embeddings.values)
    embedding_columns = [f"emb_{i}" for i in range(embedding_dim)]
    df[embedding_columns] = embeddings_array

    return df
