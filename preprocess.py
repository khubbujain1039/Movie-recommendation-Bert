import pandas as pd

movies = pd.read_csv("movies.csv")

movies["content"] = (
    movies["title"].fillna("") +
    " " +
    movies["genres"].fillna("")
)

movies.to_csv(
    "processed_movies.csv",
    index=False
)

print("Processed dataset saved.")