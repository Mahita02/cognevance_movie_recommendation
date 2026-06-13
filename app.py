import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Merge datasets
movie_ratings = pd.merge(ratings, movies, on="movieId")

# Create user-movie matrix
movie_matrix = movie_ratings.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

movie_matrix = movie_matrix.fillna(0)

# Calculate similarity
similarity = cosine_similarity(movie_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=movie_matrix.index,
    columns=movie_matrix.index
)

# Recommend movies for User 1
user_id = 1

similar_users = similarity_df[user_id].sort_values(
    ascending=False
)[1:6]

print("Top Similar Users:")
print(similar_users)

print("\nRecommended Movies For User 1:")

similar_user_ids = similar_users.index

recommended_movies = movie_ratings[
    movie_ratings['userId'].isin(similar_user_ids)
]['title'].value_counts().head(10)

print(recommended_movies)
recommended_movies.to_csv(
    "report/recommended_movies.csv"
)