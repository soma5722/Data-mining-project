import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

# Read in ratings csv file and combine into one dataframe
df_r = pd.read_csv('/content/drive/MyDrive/Movielens/ratings.csv', chunksize=1000000)

df_rc = pd.concat(df_r)

# Read in movies csv file
df_m = pd.read_csv('/content/drive/MyDrive/Movielens/movies.csv')

# Define genres and initialize empty dataframe to hold average ratings for each genre
genres = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']

for x in genres:
  df_m[x] = 0

# Fill in applicable genres in movie dataframe
for i in range(len(df_m.genres)):
    for x in df_m.genres[i].split('|'):
      if x in genres:
        df_m.loc[i, x] = 1

# Drop genres column and merge reviews column with movies column
df_m = df_m.drop(['genres'], axis=1)

df_c = df_rc.merge(df_m, on='movieId').sort_values(by=['userId'])


# Create new dataframe with overall average ratings per user
df_avg = df_c.groupby(['userId'], as_index=False)['rating'].mean()

# Add average rating per genre to dataframe
for genre in genres:
  df_gen = df_c[df_c[genre]==1] # Get all reviews belonging to given genre
  df_gen_avg = df_gen.groupby(['userId'], as_index=False)['rating'].mean() # Group by user and get mean of all ratings as new dataframe
  df_gen_avg = df_gen_avg.rename(columns={'rating':genre}) # Change column name to reflect genre
  df_avg = df_avg.merge(df_gen_avg, on='userId', how='outer') # Merge genre dataframe onto ratings dataframe

# Genres with no rating are assigned 0
df_avg = df_avg.fillna(0)

df_likes = df_avg.copy()

# Convert ratings dataframe to boolean based on whether users liked the movie
# The average of all ratings is approximately 3.5 so liking will be defined as rating 3.5 or higher
for genre in genres:
  df_likes[genre] = np.where(df_likes[genre] >= 3.5, 1, 0)


corr_matrix = df_likes.drop(['userId'], axis=1).corr()
plt.figure(figsize=[15,8])
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Drop Mystery, Adventure userId and overall rating columns to use dataframe in clustering model
df = df_likes.copy()
df = df.drop(['userId', 'rating', 'Mystery', 'Adventure'], axis=1)

# Test k-values from 2 to 30
inertias = []
clusters = []
silhouettes = []
for cluster in range(2, 31): 
    clusters.append(cluster)
    kmeans = KMeans(n_clusters = cluster, random_state = 28).fit(df)
    kmeans_pred = kmeans.predict(df)
    silhouettes.append(silhouette_score(df, kmeans_pred, sample_size=1000))
    inertias.append(kmeans.inertia_)

plt.figure(figsize=[12,8])
plt.xticks(np.arange(0, 31, 1))
plt.ylabel('Inertia')
plt.xlabel('Cluster Count (k-values)')
plt.plot(clusters, inertias)

plt.figure(figsize=[12,8])
plt.xticks(np.arange(0, 31, 1))
plt.ylabel('Average Silhouette Scores')
plt.xlabel('Cluster Count (k-values)')
plt.plot(clusters, silhouettes)

model = KMeans(12, random_state=28, n_init = 10, max_iter = 100)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(df)
visualizer.show()

model = KMeans(16, random_state=28, n_init = 10, max_iter = 100)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(df)
visualizer.show()

# Final clustering model using 16 clusters, adjusted parameters
kmeans16 = KMeans(n_clusters = 16, random_state = 42, n_init = 10, max_iter = 100).fit(df)
kmeans16_pred = kmeans16.predict(df)
silhouette = silhouette_score(df, kmeans16_pred, sample_size=1000)
inertia = kmeans16.inertia_
print('K-Means 16 Clusters')
print('silhouette score: ', silhouette)
print('inertia: ', inertia)

# Add cluster assignments to ratings dataframe containing userId to use in generating recommendations
df_avg['Cluster'] = pd.Series(kmeans16_pred, index=df_avg.index)

# Function to get recommendations of top rated movies from a given cluster
# 'cluster' is the cluster number
# 'df_model' is the dataframe used to make the model
# 'df_ratings' is the dataframe with full rating and title data
# 'min_ratings' filters out movies with less than the specified number of ratings to return only popular movies
# 'top_n' is the number of recommended movies returned
def get_top_recs(cluster, df_model, df_ratings, min_ratings, top_n):
    cluster_ids = df_model[df_model['Cluster'] == cluster]['userId'].tolist()
    df_cluster = df_ratings[df_ratings['userId'].isin(cluster_ids)]
    com_ratings = df_cluster.groupby('movieId').filter(lambda x: len(x) > min_ratings)
    return com_ratings.groupby('title').mean()['rating'].reset_index().sort_values('rating', ascending=False).head(top_n)

get_top_recs(11, df_avg, df_c, 50, 10)
