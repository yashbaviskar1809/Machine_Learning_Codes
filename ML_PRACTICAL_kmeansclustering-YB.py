#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[4]:


# create DataFrame
df = pd.read_csv('dataset.csv')
data = np.array(df)

# view first five rows of DataFrame
print(df.head())


# In[6]:


# create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaled_df = StandardScaler().fit_transform(df)

#vview first five rows of scaled DataFrame
print(scaled_df[:5])


# In[8]:


# initialize kmeans parameters
kmeans_kwargs = {
"init": "random",
"n_init": 10,
"random_state": 1,
}

# create list to hold SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

# visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[18]:


# instantiate the k-means class, using optimal number of clusters
kmeans = KMeans(init="random", n_clusters=7, n_init=10, random_state=0)
y_means = kmeans.fit_predict(data)

plt.scatter(data[y_means==0,0],data[y_means==0,1],s=50, c='brown',label='1')
plt.scatter(data[y_means==1,0],data[y_means==1,1],s=50, c='blue',label='2')
plt.scatter(data[y_means==2,0],data[y_means==2,1],s=50, c='green',label='3')
plt.scatter(data[y_means==3,0],data[y_means==3,1],s=50, c='cyan',label='4')
plt.scatter(data[y_means==4,0],data[y_means==4,1],s=50, c='orange',label='5')
plt.scatter(data[y_means==5,0],data[y_means==5,1],s=50, c='pink',label='6')
plt.scatter(data[y_means==6,0],data[y_means==6,1],s=50, c='red',label='6')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=50,marker='^', c='black', label='Centroids')
plt.title('Income Spent Analysis')
plt.xlabel('Income')
plt.ylabel('Spent')
plt.legend()
plt.show()

