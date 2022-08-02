import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

df_m = pd.read_csv('movies.csv')
df_r = pd.read_csv('ratings.csv')

df_c = df_r.merge(df_m, on='movieId')

genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

df_avg = pd.DataFrame(columns = ['userId']+genres)

for i in range(1, df_c['userId'].max()+1):
    user_ratings = df_c[df_c['userId'] == i] #get all ratings of selected user
    u_row = {'userId':i}
    for genre in genres:
        u_row[genre] = user_ratings[user_ratings['genres'].str.contains(genre)].rating.mean() #add average rating of each genre
    df_avg = df_avg.append(u_row, ignore_index=True)
    
df_avgr = df_avg.fillna(0) #fill missing data with 0 (assume no ratings means no interest)
df_avgr = df_avgr.drop('userId', axis=1)

corr_matrix = df_avgr.corr()
plt.figure(figsize=[15,8])
sns.heatmap(corr_matrix, annot=True)
plt.show()

df_avgr = df_avgr.drop(['Documentary', "Children's", 'Film-Noir'], axis=1)

df_avgsm = df_avgr.drop(['Animation', 'Fantasy', 'Horror', 'Musical', 'War', 'Western'], axis=1)

df_three = df_avgr[['Action', 'Comedy', 'Drama']]

inertias = []
sil_scores = []
clusters = []
for cluster in range(2, 26): 
    clusters.append(cluster)
    kmeans = KMeans(n_clusters = cluster, random_state = 35).fit(df_avgr)
    kmeans_pred = kmeans.predict(df_avgr)
    sil_scores.append(silhouette_score(df_avgr, kmeans_pred))
    inertias.append(kmeans.inertia_)

inertias_sm = []
sil_scores_sm = []
clusters_sm = []
for cluster in range(2, 26): 
    clusters_sm.append(cluster)
    kmeans = KMeans(n_clusters = cluster, random_state = 35).fit(df_avgsm)
    kmeans_pred = kmeans.predict(df_avgsm)
    sil_scores_sm.append(silhouette_score(df_avgsm, kmeans_pred))
    inertias_sm.append(kmeans.inertia_)
    
inertias_t = []
sil_scores_t = []
clusters_t = []
for cluster in range(2, 26): 
    clusters_t.append(cluster)
    kmeans = KMeans(n_clusters = cluster, random_state = 35).fit(df_three)
    kmeans_pred = kmeans.predict(df_three)
    sil_scores_t.append(silhouette_score(df_three, kmeans_pred))
    inertias_t.append(kmeans.inertia_)

plt.figure(figsize=[12,8])
plt.xticks(np.arange(0, 26, 1))
plt.ylabel('Inertia')
plt.xlabel('Cluster Count (k-values)')
plt.title('Elbow Plot with 15 Genres')
plt.plot(clusters, inertias)

plt.figure(figsize=[12,8])
plt.xticks(np.arange(0, 26, 1))
plt.ylabel('Inertia')
plt.xlabel('Cluster Count (k-values)')
plt.title('Elbow Plot with 9 Genres')
plt.plot(clusters_sm, inertias_sm)

plt.figure(figsize=[12,8])
plt.xticks(np.arange(0, 26, 1))
plt.ylabel('Inertia')
plt.xlabel('Cluster Count (k-values)')
plt.title('Elbow Plot with 3 Genres')
plt.plot(clusters_t, inertias_t)

plt.figure(figsize=[12,8])
plt.xticks(np.arange(0, 26, 1))
plt.ylabel('Average Silhouette Scores')
plt.xlabel('Cluster Count (k-values)')
plt.title('Silhouette Coefficients')
plt.plot(clusters_t, sil_scores_t)
plt.plot(clusters_sm, sil_scores_sm)
plt.plot(clusters, sil_scores)
plt.legend(['3 Genres', '9 Genres', '15 Genres'], loc='upper right')
