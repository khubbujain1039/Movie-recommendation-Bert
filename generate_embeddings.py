import pandas as pd
import torch
import joblib

from tqdm import tqdm
from transformers import BertTokenizer, BertModel

movies = pd.read_csv(
    "processed_movies.csv"
)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = BertModel.from_pretrained(
    "bert-base-uncased"
)

model.eval()

embeddings = []

for text in tqdm(movies["content"]):

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = (
        outputs.last_hidden_state[:, 0, :]
        .squeeze()
        .numpy()
    )

    embeddings.append(cls_embedding)

joblib.dump(
    embeddings,
    "movie_embeddings.pkl"
)

joblib.dump(
    movies["title"].tolist(),
    "movie_titles.pkl"
)

print("Embeddings saved.")