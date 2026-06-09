import joblib
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

embeddings = np.array(
    joblib.load(
        "movie_embeddings.pkl"
    )
)

movie_titles = joblib.load(
    "movie_titles.pkl"
)

def recommend(movie_name, top_n=10):

    if movie_name not in movie_titles:
        return ["Movie not found"]

    movie_index = movie_titles.index(
        movie_name
    )

    similarity_scores = cosine_similarity(
        [embeddings[movie_index]],
        embeddings
    )[0]

    similar_movies = sorted(
        list(enumerate(similarity_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in similar_movies[1:top_n+1]:
        recommendations.append(
            movie_titles[i[0]]
        )

    return recommendations

if __name__ == "__main__":

    movie = input("Enter movie name: ")

    recommendations = recommend(movie)

    print("\nRecommendations:")

    for rec in recommendations:
        print(rec)