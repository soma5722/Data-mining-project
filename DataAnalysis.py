import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Data Sets/movies.csv')
#print(df.nunique(axis=0))
#print(df['movieId'].nunique())
movieId = df['movieId'].nunique()
genre = df['genres'].nunique()

df = pd.read_csv('Data Sets/ratings.csv')
#print(df.nunique(axis=0))
userId = df['userId'].nunique()
ratings = df['rating'].nunique()

plt.figure(figsize = (10,7))
plt.xlabel("unique movieId")
plt.ylabel("review count")
plt.title("movies with most review - top 30")
df['movieId'].value_counts()[:30].plot(kind = 'bar')




df = pd.read_csv('Data Sets/tags.csv')
#print(df.nunique(axis=0))
tag = df['tag'].nunique()
x=['movieId','genres','userId','rating','tag']
y = [movieId,genre,userId,ratings,tag]
print(x)
print(y)
  
fig = plt.figure(figsize = (10, 5))

plt.barh(x, y)
 
for index, value in enumerate(y):
    plt.text(value, index,
             str(value)) 
# creating the bar plot
#plt.bar(x, y, color ='maroon',width = 0.4)
 
plt.ylabel("Unique attributes")
plt.xlabel("Count")
plt.title("Unique attributes vs Count")
plt.show()

#sns.barplot(x=['movieId','genres','userId','rating','tag'], y = [movieId,genre,userId,ratings,tag],palette="Blues_d")    
